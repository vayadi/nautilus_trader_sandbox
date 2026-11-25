"""Download ~10 years of daily interest rates for HUF, EUR, USD, JPY, GBP, CHF, AUD into one panel.

Output: <out-dir>/rates_<source>_<tenor>.parquet (+ .csv and a .json with the series map), one row
per business day, one column per currency, values in percent. Missing days are forward-filled
(max --ffill-limit days) so the VAR sees a dense panel.

Sources (--source) and what each one gives per currency
------------------------------------------------------
official   (recommended)  free, no key
           tenor 10y  USD  FRED DGS10                  10Y Treasury constant maturity
                      EUR  ECB YC ...SR_10Y            10Y euro-area AAA govt curve
                      GBP  BoE IUDMNPY                 10Y gilt nominal par yield
                      JPY  MoF jgbcme 10Y              10Y JGB
                      CHF  SNB rendoblim 10J           10Y Confederation bond
                      AUD  RBA F2 FCMYGBAG10D          10Y ACGB
                      HUF  ÁKK reference yield 10Y     10Y HGB benchmark (akk.hu Referenciahozamok)
           tenor short USD FRED DGS3MO, EUR ECB YC SR_3M, GBP BoE IUDSOIA (SONIA),
                      JPY MoF 1Y JGB, CHF SNB SARON, AUD RBA F1 FIRMMBAB90D (90d bank bill),
                      HUF  ÁKK reference yield 3M (3M DKJ T-bill)
yfinance   tenor short USD ^IRX (13w T-bill); EUR/JPY/GBP/CHF/AUD = covered-interest-parity
                      implied 3M rate from Yahoo spot (EURUSD=X ...) vs front CME future (6E=F ...)
                      and ^IRX. Noisy (spot and futures closes are not synchronous), --smooth helps.
           tenor 10y  USD ^TNX only.
           HUF and anything else Yahoo lacks -> --fallback source.
databento  tenor short (GLBX.MDP3, historical, needs DATABENTO_API_KEY)
                      USD  ZQ fed-funds futures: average implied rate over the months between the
                           front and second FX-futures expiries (a ~3M forward USD rate)
                      EUR/JPY/GBP/CHF/AUD  CIP-implied 3M forward rate from the 6E/6J/6B/6S/6A
                           front vs second quarterly futures (calendar spread) and the USD rate
                      HUF  same from the CME HUF/USD futures (root --huf-root, default HUF); these are
                           thinly traded, so check the output. If no data -> --fallback.
           tenor 10y  not supported (no 10Y futures for most of these on CME) -> use official.

akk        HUF only: ÁKK reference yields (daily since 2016-01-04; 3M 6M 12M 3Y 5Y 10Y 15Y 20Y, 20Y from
           2020-05). JSON API https://www.akk.hu/api/data/xpose3-dwh/data?query=getBenchmark; raw JSON is
           cached per year in <out-dir>/raw/akk/, closed years are not re-fetched (--akk-refresh forces it).
           Tenor 10y -> 10Y, short -> 3M, or pick one with --akk-tenor 5Y. Other currencies -> --fallback.
           --akk-dump writes all tenors to <out-dir>/akk_benchmark_yields.csv (+ _long.csv with the
           benchmark papers' ISIN, bid/ask, price) and exits.

A per-currency override is possible: --override HUF=akk JPY=yfinance, or --csv CCY=path.csv
(any CSV with a date column and one value column, in percent; --csv wins over everything).

  python rates_VectorAutoRegression_data.py --source official --tenor 10y
  python rates_VectorAutoRegression_data.py --source official --tenor 10y --override HUF=akk --akk-tenor 5Y
  python rates_VectorAutoRegression_data.py --akk-dump
  python rates_VectorAutoRegression_data.py --source databento --tenor short --override HUF=akk --max-cost 5
  python rates_VectorAutoRegression_data.py --source yfinance --tenor short --fallback official
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

CCYS = ["HUF", "EUR", "USD", "JPY", "GBP", "CHF", "AUD"]
FX_ROOT = {"EUR": "6E", "JPY": "6J", "GBP": "6B", "CHF": "6S", "AUD": "6A"}  # all USD per 1 unit ccy
DAYCOUNT = {"EUR": 360, "JPY": 360, "CHF": 360, "USD": 360, "HUF": 360, "GBP": 365, "AUD": 365}
# BoE returns 403 without this UA; RBA (403) and FRED (hangs) reject it, so only boe() sends it.
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) rates-downloader/1.0"}


# ----------------------------------------------------------------------------------------- utils
def _get(url: str, timeout: int = 60, headers: dict | None = None) -> bytes:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _series(dates, values, name: str) -> pl.DataFrame:
    df = pl.DataFrame({"date": dates, name: values})
    df = df.with_columns(pl.col("date").cast(pl.Date), pl.col(name).cast(pl.Float64, strict=False))
    return df.drop_nulls().unique("date", keep="last").sort("date")


def _imm(y: int, m: int) -> date:
    """Third Wednesday of month m (IMM date)."""
    d = date(y, m, 15)
    return d + timedelta(days=(2 - d.weekday()) % 7)


# -------------------------------------------------------------------------------------- official
def fred(series: str, start: date) -> tuple[list, list]:
    raw = _get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd={start}")
    df = pl.read_csv(io.BytesIO(raw), infer_schema_length=0)
    return df[df.columns[0]].str.to_date("%Y-%m-%d").to_list(), df[series].to_list()


def ecb_yc(tenor: str, start: date) -> tuple[list, list]:
    key = f"B.U2.EUR.4F.G_N_A.SV_C_YM.SR_{tenor}"
    raw = _get(f"https://data-api.ecb.europa.eu/service/data/YC/{key}?format=csvdata&startPeriod={start}")
    df = pl.read_csv(io.BytesIO(raw), infer_schema_length=0)
    return df["TIME_PERIOD"].str.to_date("%Y-%m-%d").to_list(), df["OBS_VALUE"].to_list()


def boe(series: str, start: date) -> tuple[list, list]:
    url = ("https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp?csv.x=yes"
           f"&Datefrom={start:%d/%b/%Y}&Dateto=now&SeriesCodes={series}&CSVF=TN&UsingCodes=Y&VPD=Y&VFD=N")
    df = pl.read_csv(io.BytesIO(_get(url, headers=UA)), infer_schema_length=0)
    return df[df.columns[0]].str.to_date("%d %b %Y").to_list(), df[series].to_list()


def snb(cube: str, d0: str, start: date, d1: str | None = None) -> tuple[list, list]:
    txt = _get(f"https://data.snb.ch/api/cube/{cube}/data/csv/en").decode("utf-8-sig")
    lines = txt.splitlines()
    i = next(k for k, l in enumerate(lines) if l.replace('"', "").startswith("Date;"))
    df = pl.read_csv(io.StringIO("\n".join(lines[i:])), separator=";", infer_schema_length=0)
    df = df.filter(pl.col("D0") == d0)
    if d1 is not None:
        df = df.filter(pl.col("D1") == d1)
    d = df["Date"].str.to_date("%Y-%m-%d", strict=False)
    if d.null_count() == len(d):                      # monthly cubes use YYYY-MM
        d = (df["Date"] + "-01").str.to_date("%Y-%m-%d")
    keep = [x >= start if x else False for x in d.to_list()]
    return [x for x, k in zip(d.to_list(), keep) if k], [v for v, k in zip(df["Value"].to_list(), keep) if k]


def rba(table: str, series_id: str, start: date) -> tuple[list, list]:
    txt = _get(f"https://www.rba.gov.au/statistics/tables/csv/{table}-data.csv").decode("latin-1")
    lines = txt.splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith("Series ID"))
    df = pl.read_csv(io.StringIO("\n".join(lines[i:])), infer_schema_length=0)
    df = df.rename({df.columns[0]: "date"})
    d = df["date"].str.to_date("%d-%b-%Y", strict=False)
    keep = [x is not None and x >= start for x in d.to_list()]
    return [x for x, k in zip(d.to_list(), keep) if k], [v for v, k in zip(df[series_id].to_list(), keep) if k]


def mof_jgb(col: str, start: date) -> tuple[list, list]:
    base = "https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/"
    out_d, out_v = [], []
    for url in (base + "historical/jgbcme_all.csv", base + "jgbcme.csv"):
        lines = _get(url).decode("utf-8", "replace").splitlines()
        i = next(k for k, l in enumerate(lines) if l.startswith("Date"))
        df = pl.read_csv(io.StringIO("\n".join(lines[i:])), infer_schema_length=0)
        d = df["Date"].str.to_date("%Y/%m/%d", strict=False)
        for x, v in zip(d.to_list(), df[col].to_list()):
            if x is not None and x >= start:
                out_d.append(x)
                out_v.append(None if v in ("-", "") else v)
    return out_d, out_v


# ------------------------------------------------------------------------------------------- ÁKK
# Hungarian Government Debt Management Agency, Statisztika > Referenciahozamok. The page is fed by
#   GET https://www.akk.hu/api/data/xpose3-dwh/data?query=getBenchmark&dateStart=YYYY-MM-DD&dateEnd=...
# (JSON; `...&help` lists the fields). One row per fixing date and tenor: date, tenor, yield (the
# reference yield, mid of avgBid/avgAsk, in %), name/isin of the benchmark paper, netPrice, ...
# History starts 2016-01-04 (20Y from 2020-05-20). A 10-year range is one ~6 MB call, but raw JSON is
# cached per calendar year under <out-dir>/raw/akk/ and only years that are not closed are re-fetched.
# (Eurostat's daily HU 10Y irt_lt_mcby_d, used before, was withdrawn in 2026-11.)
AKK_URL = "https://www.akk.hu/api/data/xpose3-dwh/data?query=getBenchmark&dateStart={0}&dateEnd={1}"
AKK_TENORS = ["3M", "6M", "12M", "3Y", "5Y", "10Y", "15Y", "20Y"]
AKK_FIRST = date(2016, 1, 4)
AKK_DIR = Path("rates_VectorAutoRegression/data/raw/akk")    # main() points this at <out-dir>/raw/akk
AKK_TENOR = {"10y": "10Y", "short": "3M"}                   # panel tenor -> ÁKK tenor (--akk-tenor overrides)
_akk_mem: dict[int, list[dict]] = {}


def _akk_year(y: int, cache_dir: Path, refresh: bool) -> list[dict]:
    """Raw ÁKK benchmark rows for calendar year y; closed years come from the cache once fetched."""
    if y in _akk_mem and not refresh:
        return _akk_mem[y]
    f = cache_dir / f"benchmark_{y}.json"
    closed = date(y, 12, 31) < date.today() - timedelta(days=7)   # a week of slack for late fixings
    if f.exists() and closed and not refresh:
        rows = json.loads(f.read_text())["data"]
    else:
        d0, d1 = max(date(y, 1, 1), AKK_FIRST), min(date(y, 12, 31), date.today())
        raw = json.loads(_get(AKK_URL.format(d0, d1), timeout=120, headers=UA))
        if raw.get("meta", {}).get("status") != "OK":
            raise RuntimeError(f"ÁKK getBenchmark {y}: {raw.get('meta')}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(raw))
        rows = raw["data"]
        print(f"  akk      fetched {y}: {len(rows)} rows -> {f}")
    _akk_mem[y] = rows
    return rows


def akk_rows(start: date, end: date, cache_dir: Path | None = None, refresh: bool = False) -> list[dict]:
    cache_dir = Path(cache_dir or AKK_DIR)
    s, e = str(max(start, AKK_FIRST)), str(end)
    out = []
    for y in range(max(start, AKK_FIRST).year, end.year + 1):
        out += [r for r in _akk_year(y, cache_dir, refresh) if s <= r["date"] <= e]
    return out


def akk_yields(tenor: str, start: date, end: date | None = None, cache_dir: Path | None = None,
               refresh: bool = False) -> tuple[list, list]:
    """Daily ÁKK reference yield (%) for tenor in AKK_TENORS (e.g. '10Y', '3M'): (dates, values)."""
    tenor = tenor.upper()
    if tenor not in AKK_TENORS:
        raise ValueError(f"ÁKK tenor {tenor!r} not in {AKK_TENORS}")
    rows = [r for r in akk_rows(start, end or date.today(), cache_dir, refresh)
            if r["tenor"] == tenor and r.get("yield") is not None]
    rows.sort(key=lambda r: r["date"])
    return [date.fromisoformat(r["date"]) for r in rows], [r["yield"] for r in rows]


def akk_dump(path: Path, start: date, end: date, cache_dir: Path | None = None, refresh: bool = False) -> pl.DataFrame:
    """All ÁKK tenors as a wide CSV: date, 3M, 6M, ..., 20Y (yields in %), plus a long CSV with the papers."""
    rows = akk_rows(start, end, cache_dir, refresh)
    cols = ["date", "tenor", "yield", "avgBid", "avgAsk", "netPrice", "name", "isin", "maturityDate", "securityType"]
    long = (pl.DataFrame([{k: r.get(k) for k in cols} for r in rows], infer_schema_length=None)
            .with_columns(pl.col("date").str.to_date(), pl.col("yield", "avgBid", "avgAsk", "netPrice").cast(pl.Float64))
            .sort("date", "tenor"))
    wide = long.pivot(on="tenor", index="date", values="yield", aggregate_function="last").sort("date")
    wide = wide.select("date", *[t for t in AKK_TENORS if t in wide.columns])
    path.parent.mkdir(parents=True, exist_ok=True)
    wide.write_csv(path)
    long.write_csv(path.with_name(path.stem + "_long.csv"))
    return wide


OFFICIAL = {
    "10y": {
        "USD": ("FRED DGS10", lambda s: fred("DGS10", s)),
        "EUR": ("ECB YC SR_10Y (AAA)", lambda s: ecb_yc("10Y", s)),
        "GBP": ("BoE IUDMNPY", lambda s: boe("IUDMNPY", s)),
        "JPY": ("MoF JGB 10Y", lambda s: mof_jgb("10Y", s)),
        # rendoblim (monthly) / rendoblid stopped 2025-07; rendeiduebd = daily spot rates, published monthly
        "CHF": ("SNB rendeiduebd CHF 10J (spot)", lambda s: snb("rendeiduebd", "CHF", s, "10J")),
        "AUD": ("RBA F2 FCMYGBAG10D", lambda s: rba("f2", "FCMYGBAG10D", s)),
        "HUF": ("ÁKK reference yield 10Y", lambda s: akk_yields("10Y", s)),
    },
    "short": {
        "USD": ("FRED DGS3MO", lambda s: fred("DGS3MO", s)),
        "EUR": ("ECB YC SR_3M (AAA)", lambda s: ecb_yc("3M", s)),
        "GBP": ("BoE IUDSOIA (SONIA)", lambda s: boe("IUDSOIA", s)),
        "JPY": ("MoF JGB 1Y", lambda s: mof_jgb("1Y", s)),
        "CHF": ("SNB snbgwdzid SARON", lambda s: snb("snbgwdzid", "SARON", s)),
        "AUD": ("RBA F1 FIRMMBAB90D", lambda s: rba("f1", "FIRMMBAB90D", s)),
        "HUF": ("ÁKK reference yield 3M", lambda s: akk_yields("3M", s)),
    },
}


def load_official(ccys, tenor, start, end, **_) -> dict[str, tuple[str, pl.DataFrame]]:
    table = OFFICIAL[tenor]
    out = {}

    def one(c):
        label, fn = table[c]
        d, v = fn(start)
        return c, label, _series(d, v, c).filter(pl.col("date") <= end)

    with ThreadPoolExecutor(max_workers=7) as ex:
        futs = {c: ex.submit(one, c) for c in ccys if c in table}
        for c, f in futs.items():
            try:
                c, label, df = f.result()
                out[c] = (label, df)
                print(f"  official {c:<3} {label:<28} {df.height:>5} rows")
            except Exception as e:  # noqa: BLE001
                print(f"  official {c:<3} FAILED: {e!r}")
    return out


# -------------------------------------------------------------------------------------- yfinance
YF_SPOT = {"EUR": ("EURUSD=X", False), "GBP": ("GBPUSD=X", False), "AUD": ("AUDUSD=X", False),
           "JPY": ("JPY=X", True), "CHF": ("CHF=X", True)}          # True: quoted ccy per USD


def _yf_close(tickers: list[str], start: date, end: date) -> pl.DataFrame:
    import yfinance as yf
    raw = yf.download(tickers, start=str(start), end=str(end + timedelta(days=1)), progress=False,
                      auto_adjust=False, threads=True)
    close = raw["Close"] if "Close" in raw else raw
    close = close.reset_index()
    close.columns = [str(c) for c in close.columns]
    df = pl.from_pandas(close).rename({close.columns[0]: "date"})
    return df.with_columns(pl.col("date").cast(pl.Date))


def _front_tau(d: date, min_days: int) -> tuple[float | None, int]:
    """Days to the front quarterly IMM date (CME FX futures value date) as year fraction/360."""
    for k in range(0, 5):
        m0 = ((d.month - 1) // 3) * 3 + 3 + 3 * k        # next Mar/Jun/Sep/Dec
        y = d.year + (m0 - 1) // 12
        m = (m0 - 1) % 12 + 1
        e = _imm(y, m)
        if e > d:
            days = (e - d).days
            return (days / 360.0 if days >= min_days else None), days
    return None, 0


def load_yfinance(ccys, tenor, start, end, smooth=5, min_days=15, **_):
    out = {}
    usd_t = "^IRX" if tenor == "short" else "^TNX"
    need = [c for c in ccys if c != "HUF"]
    tickers = [usd_t]
    if tenor == "short":
        for c in need:
            if c in YF_SPOT:
                tickers += [YF_SPOT[c][0], f"{FX_ROOT[c]}=F"]
    px = _yf_close(tickers, start, end)
    usd = px.select("date", pl.col(usd_t).alias("USD")).drop_nulls()
    if "USD" in ccys and usd.height:
        out["USD"] = (f"Yahoo {usd_t}", usd)
        print(f"  yfinance USD {usd_t:<20} {usd.height:>5} rows")
    if tenor != "short":
        return out
    for c in need:
        if c not in YF_SPOT:
            continue
        spot_t, inverted = YF_SPOT[c]
        fut_t = f"{FX_ROOT[c]}=F"
        df = px.select("date", pl.col(usd_t).alias("r"), pl.col(spot_t).alias("s"), pl.col(fut_t).alias("f"))
        df = df.drop_nulls()
        if df.height == 0:
            print(f"  yfinance {c} no data for {spot_t}/{fut_t}")
            continue
        s = df["s"].to_numpy()
        s = 1.0 / s if inverted else s
        f = df["f"].to_numpy()
        r = df["r"].to_numpy() / 100.0
        tau = np.array([_front_tau(d, min_days)[0] or np.nan for d in df["date"].to_list()])
        # F = S (1 + r_usd T) / (1 + r_ccy T)  (USD per 1 ccy)  ->  r_ccy = ((1 + r_usd T) S/F - 1)/T
        rc = ((1 + r * tau) * s / f - 1) / tau * 100.0
        ser = pl.DataFrame({"date": df["date"], c: rc}).with_columns(pl.col(c).fill_nan(None))
        if smooth and smooth > 1:
            ser = ser.with_columns(pl.col(c).forward_fill().ewm_mean(span=smooth, ignore_nulls=True))
        ser = ser.drop_nulls()
        out[c] = (f"CIP {spot_t} vs {fut_t} + {usd_t}" + (f" (EWMA {smooth})" if smooth > 1 else ""), ser)
        print(f"  yfinance {c:<3} CIP {spot_t}/{fut_t:<10} {ser.height:>5} rows")
    return out


# ------------------------------------------------------------------------------------- databento
def _db_client():
    import databento as db
    if not os.environ.get("DATABENTO_API_KEY"):
        sys.exit("DATABENTO_API_KEY is not set")
    return db.Historical()


def _db_query(client, symbols, schema, start, end, stype_in, max_cost, dry_run):
    kw = dict(dataset="GLBX.MDP3", symbols=symbols, schema=schema, stype_in=stype_in,
              start=str(start), end=str(end + timedelta(days=1)))
    cost = client.metadata.get_cost(**kw)
    print(f"  databento {schema:<10} {len(symbols):>3} symbols  est. cost ${cost:.2f}")
    if dry_run:
        return None, cost
    if max_cost is not None and cost > max_cost:
        sys.exit(f"estimated cost ${cost:.2f} > --max-cost {max_cost}")
    return client.timeseries.get_range(**kw).to_df().reset_index(), cost


def load_databento(ccys, tenor, start, end, huf_root="HUF", max_cost=None, dry_run=False, **_):
    if tenor != "short":
        print("  databento: only --tenor short is supported (futures-implied 3M rates)")
        return {}
    client = _db_client()
    roots = {c: FX_ROOT[c] for c in ccys if c in FX_ROOT}
    if "HUF" in ccys and huf_root:
        roots["HUF"] = huf_root
    syms = [f"{r}.c.{k}" for r in roots.values() for k in (0, 1)] + [f"ZQ.c.{k}" for k in range(8)]
    bars, cost = _db_query(client, syms, "ohlcv-1d", start, end, "continuous", max_cost, dry_run)
    ids = None if bars is None else sorted({int(i) for i in bars["instrument_id"].unique()})
    if dry_run:
        print(f"  databento dry run: bars ${cost:.2f} (+ definitions for the contracts seen, usually cents)")
        return {}
    defs, _ = _db_query(client, [str(i) for i in ids], "definition", start, end, "instrument_id", max_cost, False)
    exp = (pl.from_pandas(defs[["instrument_id", "expiration", "raw_symbol"]].astype({"expiration": "datetime64[ns, UTC]"}))
           .with_columns(pl.col("expiration").dt.date().alias("exp"))
           .group_by("instrument_id").agg(pl.col("exp").last(), pl.col("raw_symbol").last()))
    b = pl.from_pandas(bars[["ts_event", "instrument_id", "symbol", "close"]].astype({"ts_event": "datetime64[ns, UTC]"}))
    b = (b.with_columns(pl.col("ts_event").dt.date().alias("date"), pl.col("instrument_id").cast(pl.Int64))
          .join(exp.with_columns(pl.col("instrument_id").cast(pl.Int64)), on="instrument_id", how="left"))

    # USD: ZQ monthly futures -> implied average EFFR per calendar month (100 - price)
    zq = (b.filter(pl.col("symbol").str.starts_with("ZQ."))
            .with_columns((100 - pl.col("close")).alias("rate"),
                          pl.col("exp").dt.month_start().alias("month")))
    zq_map: dict[date, dict[date, float]] = {}
    for d, m, r in zq.select("date", "month", "rate").iter_rows():
        if m is not None:
            zq_map.setdefault(d, {})[m] = r

    def usd_fwd(d: date, e1: date, e2: date) -> float | None:
        """Day-weighted mean ZQ-implied rate over [e1, e2)."""
        curve = zq_map.get(d)
        if not curve or e1 is None or e2 is None or e2 <= e1:
            return None
        tot, w = 0.0, 0
        x = e1
        while x < e2:
            m = x.replace(day=1)
            nxt = (m + timedelta(days=32)).replace(day=1)
            seg = (min(nxt, e2) - x).days
            if m not in curve:
                return None
            tot += curve[m] * seg
            w += seg
            x = min(nxt, e2)
        return tot / w

    out = {}
    usd_rows = {}
    for c, root in roots.items():
        legs = b.filter(pl.col("symbol").is_in([f"{root}.c.0", f"{root}.c.1"]))
        if legs.height == 0:
            print(f"  databento {c} ({root}): no bars")
            continue
        w = (legs.with_columns(pl.col("symbol").str.slice(-1).alias("leg"))
                 .pivot(on="leg", index="date", values=["close", "exp"], aggregate_function="last")
                 .drop_nulls().sort("date"))
        dates, vals = [], []
        for d, f1, f2, e1, e2 in w.select("date", "close_0", "close_1", "exp_0", "exp_1").iter_rows():
            ru = usd_fwd(d, e1, e2)
            if ru is None or not f1 or not f2:
                continue
            tau = (e2 - e1).days / 360.0
            ru = ru / 100.0
            # forward-forward CIP between the two expiries: F2/F1 = (1+r_usd tau)/(1+r_ccy tau)
            rc = ((1 + ru * tau) * f1 / f2 - 1) / tau
            dates.append(d)
            vals.append(rc * 100.0)
            usd_rows.setdefault(d, ru * 100.0)
        ser = _series(dates, vals, c)
        out[c] = (f"CIP {root}.c.0/{root}.c.1 + ZQ (GLBX)", ser)
        print(f"  databento {c:<3} {root:<4} {ser.height:>5} rows")
    if "USD" in ccys and usd_rows:
        k = sorted(usd_rows)
        out["USD"] = ("ZQ implied fwd over FX futures window (GLBX)", _series(k, [usd_rows[x] for x in k], "USD"))
        print(f"  databento USD ZQ   {len(k):>5} rows")
    return out


# ------------------------------------------------------------------------------------------- csv
def load_csv(ccy: str, path: str, start: date, end: date) -> tuple[str, pl.DataFrame]:
    df = pl.read_csv(path, infer_schema_length=0, try_parse_dates=False)
    dcol = next((c for c in df.columns if c.lower() in ("date", "datum", "dátum", "time_period")), df.columns[0])
    vcol = next(c for c in df.columns if c != dcol)
    d = None
    for fmt in ("%Y-%m-%d", "%Y.%m.%d", "%Y.%m.%d.", "%d/%m/%Y", "%m/%d/%Y", "%d.%m.%Y", "%Y/%m/%d"):
        d = df[dcol].str.strip_chars().str.to_date(fmt, strict=False)
        if d.null_count() < len(d) * 0.1:
            break
    v = df[vcol].str.replace(",", ".").str.strip_chars()
    ser = _series(d.to_list(), v.to_list(), ccy).filter(pl.col("date").is_between(start, end))
    print(f"  csv      {ccy:<3} {path:<28} {ser.height:>5} rows")
    return f"CSV {Path(path).name}:{vcol}", ser


def load_akk(ccys, tenor, start, end, akk_tenor=None, **_):
    """--source/--override akk: HUF only, ÁKK reference yield (10y -> 10Y, short -> 3M, or --akk-tenor)."""
    if "HUF" not in ccys:
        return {}
    t = (akk_tenor or AKK_TENOR[tenor]).upper()
    d, v = akk_yields(t, start, end)
    ser = _series(d, v, "HUF")
    print(f"  akk      HUF ÁKK {t:<24} {ser.height:>5} rows")
    return {"HUF": (f"ÁKK reference yield {t}", ser)}


LOADERS = {"official": load_official, "yfinance": load_yfinance, "databento": load_databento, "akk": load_akk}


# ------------------------------------------------------------------------------------------ main
def build_panel(series: dict[str, tuple[str, pl.DataFrame]], start, end, ffill_limit: int) -> pl.DataFrame:
    cal = pl.DataFrame({"date": pl.date_range(start, end, "1d", eager=True)})
    cal = cal.filter(pl.col("date").dt.weekday() <= 5)
    for c in CCYS:
        if c in series:
            cal = cal.join(series[c][1], on="date", how="left")
    cols = [c for c in CCYS if c in cal.columns]
    cal = cal.with_columns([pl.col(c).forward_fill(limit=ffill_limit) for c in cols])
    return cal.drop_nulls()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=list(LOADERS), default="official")
    ap.add_argument("--tenor", choices=["10y", "short"], default=None,
                    help="default: 10y for official, short for yfinance/databento")
    ap.add_argument("--years", type=float, default=10.0)
    ap.add_argument("--start", type=date.fromisoformat)
    ap.add_argument("--end", type=date.fromisoformat, default=date.today() - timedelta(days=1))
    ap.add_argument("--ccys", nargs="+", default=CCYS)
    ap.add_argument("--fallback", choices=list(LOADERS) + ["none"], default="official",
                    help="source for currencies the main source lacks")
    ap.add_argument("--override", nargs="*", default=[], metavar="CCY=SOURCE")
    ap.add_argument("--csv", nargs="*", default=[], metavar="CCY=PATH")
    ap.add_argument("--smooth", type=int, default=5, help="EWMA span for yfinance CIP rates (1 = off)")
    ap.add_argument("--huf-root", default="HUF", help="Databento GLBX root of the HUF/USD future ('' = skip)")
    ap.add_argument("--max-cost", type=float, default=None, help="abort Databento request above this $")
    ap.add_argument("--dry-run", action="store_true", help="Databento: print cost only")
    ap.add_argument("--ffill-limit", type=int, default=5)
    ap.add_argument("--out-dir", default="rates_VectorAutoRegression/data")
    ap.add_argument("--akk-tenor", choices=AKK_TENORS, default=None,
                    help="ÁKK tenor for HUF with --override HUF=akk / --source akk (default 10Y or 3M)")
    ap.add_argument("--akk-dump", action="store_true",
                    help="only write all ÁKK tenors to <out-dir>/akk_benchmark_yields.csv (+ _long.csv) and exit")
    ap.add_argument("--akk-refresh", action="store_true", help="re-fetch every ÁKK year, ignore the cache")
    a = ap.parse_args(argv)

    global AKK_DIR
    AKK_DIR = Path(a.out_dir) / "raw" / "akk"
    if a.akk_refresh:
        for y in range(AKK_FIRST.year, date.today().year + 1):
            _akk_year(y, AKK_DIR, refresh=True)
    tenor = a.tenor or ("10y" if a.source in ("official", "akk") else "short")
    start = a.start or (a.end - timedelta(days=int(a.years * 365.25)))
    if a.akk_dump:
        path = Path(a.out_dir) / "akk_benchmark_yields.csv"
        wide = akk_dump(path, a.start or AKK_FIRST, a.end)
        print(f"\nÁKK {wide.height} days {wide['date'].min()} .. {wide['date'].max()} -> {path}")
        for t in wide.columns[1:]:
            s = wide[t].drop_nulls()
            print(f"  {t:<4} {s.len():>5} rows  last {s[-1]:6.3f}  min {s.min():6.3f}  max {s.max():6.3f}")
        return
    ccys = [c.upper() for c in a.ccys]
    if "HUF" not in ccys:
        sys.exit("HUF must be in --ccys (it is the target)")
    override = dict(x.upper().split("=", 1) for x in a.override)
    override = {k: v.lower() for k, v in override.items()}
    csvs = {k.upper(): v for k, v in (x.split("=", 1) for x in a.csv)}
    print(f"{a.source} / tenor {tenor} / {start} .. {a.end}")

    got: dict[str, tuple[str, pl.DataFrame]] = {}
    for c, p in csvs.items():
        got[c] = load_csv(c, p, start, a.end)
    by_src: dict[str, list[str]] = {}
    for c in ccys:
        if c not in got:
            by_src.setdefault(override.get(c, a.source), []).append(c)
    kw = dict(start=start, end=a.end, smooth=a.smooth, huf_root=a.huf_root, max_cost=a.max_cost, dry_run=a.dry_run,
              akk_tenor=a.akk_tenor)
    for src, cs in by_src.items():
        got.update(LOADERS[src](cs, tenor, **kw))
    if a.dry_run:
        return
    missing = [c for c in ccys if c not in got or got[c][1].height == 0]
    if missing and a.fallback != "none" and set(by_src) != {a.fallback}:
        print(f"fallback {a.fallback} for {missing}")
        got.update({k: v for k, v in LOADERS[a.fallback](missing, tenor, **kw).items() if v[1].height})
    missing = [c for c in ccys if c not in got or got[c][1].height == 0]
    if "HUF" in missing:
        sys.exit("no HUF series (ÁKK download failed?): try --akk-refresh, or pass --csv HUF=<csv>")
    if missing:
        print(f"WARNING: dropping currencies with no data: {missing}")

    panel = build_panel({c: v for c, v in got.items() if c in ccys}, start, a.end, a.ffill_limit)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"rates_{a.source}_{tenor}"
    panel.write_parquet(stem.with_suffix(".parquet"))
    panel.write_csv(stem.with_suffix(".csv"))
    meta = {"source": a.source, "tenor": tenor, "start": str(start), "end": str(a.end), "rows": panel.height,
            "series": {c: got[c][0] for c in panel.columns if c != "date"}}
    stem.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"\n{panel.height} aligned business days {panel['date'].min()} .. {panel['date'].max()} -> {stem}.parquet")
    for c, lab in meta["series"].items():
        s = panel[c]
        print(f"  {c:<3} {lab:<45} last {s[-1]:6.3f}  min {s.min():6.3f}  max {s.max():6.3f}")


if __name__ == "__main__":
    main()
