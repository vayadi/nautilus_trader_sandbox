#!/usr/bin/env python3
"""
Gamma exposure (GEX) by strike for CME futures options (Databento GLBX.MDP3) -> call/put walls.

  python futures_option_gex.py CL 2026-11-20 --dry-run          # cost estimate only
  python futures_option_gex.py CL 2026-11-20 --show             # WTI crude (LO + weeklies)
  python futures_option_gex.py GC 2026-11-20 --show             # gold (OG + weeklies)
  python futures_option_gex.py GC 2026-11-20 --parents OG.OPT   # only the standard monthly gold options
  python futures_option_gex.py CL 2026-11-20 --expiry 2026-12-16 --walls 3

Roots: the futures root, e.g. CL (crude), GC (gold), SI (silver), NG (nat gas), HG (copper), ES.
The option products on that root (monthlies and weeklies, e.g. LO, OG, and their weekly
codes) are found automatically from that day's definitions, like the downloader's `discover`.

Pipeline (raw DBN files cached under fut_opt_data/<ROOT>/<date>/, reruns are free):
  1. definition  ALL_SYMBOLS              -> option products whose underlying is a <ROOT> future
  2. statistics  <product>.OPT (parent)   -> open interest (stat_type 9, CME's prior-day OI)
                                             and settlement price (stat_type 3) per option
  3. statistics  <ROOT>.FUT               -> settlement of each underlying future (F)

IV per option from its settlement price (Black-76, OTM side), with a per-expiry smile fit for
strikes whose settlement is too small to invert. Dollar GEX per 1% move of the underlying:

    GEX = gamma * OI * multiplier * F^2 * 0.01     dealers long calls (+), short puts (-)

Plots and outputs are the same as equity_option_gex.py: gex_<ROOT>_<date>.png / .html / .csv.

Licensing: without a CME live licence Databento serves this data only about a day after it
happens, so run it the day after the trade date (the error message names the cut-off).

Requires: databento, numpy, pandas, matplotlib, plotly (conda env `nautilus`); env DATABENTO_API_KEY.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from equity_option_gex import (
    _d1, aggregate_by_strike, fit_smile, fmt_money, implied_vol, norm_pdf, plot_html, plot_png,
    wall_expiry, zero_gamma,
)

NY = ZoneInfo("America/New_York")
DATASET = "GLBX.MDP3"
MONTH_CODES = "FGHJKMNQUVXZ"
OI_STAT, SETTLE_STAT = 9, 3          # databento StatType.OPEN_INTEREST / SETTLEMENT_PRICE
# dollars per 1.0 move in the futures price (one contract)
MULTIPLIERS = {"CL": 1000, "GC": 100, "SI": 5000, "NG": 10000, "HG": 25000, "PL": 50, "PA": 100,
               "HO": 42000, "RB": 42000, "BZ": 1000, "MCL": 100, "MGC": 10,
               "ES": 50, "NQ": 20, "RTY": 50, "YM": 5, "6E": 125000, "ZN": 1000, "ZB": 1000}
# settlement time (ET) used as "now" when measuring time to expiry
SETTLE_ET = {"CL": (14, 30), "GC": (13, 30), "SI": (13, 25), "NG": (14, 30), "HG": (13, 0),
             "ES": (16, 0), "NQ": (16, 0)}


# ====================================================================== download
def client():
    import databento as db

    if not os.environ.get("DATABENTO_API_KEY"):
        sys.exit("DATABENTO_API_KEY is not set")
    return db.Historical()


def window(day: date):
    """Trade-date window in UTC: midnight -> 17:00 ET (CME daily close, after settlements)."""
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    end = datetime(day.year, day.month, day.day, 17, 0, tzinfo=NY).astimezone(timezone.utc)
    return start.isoformat(), end.isoformat()


def fetch(c, path: Path, dry_run, **kw):
    """-> cost; downloads unless dry_run (atomic rename, cached)."""
    if path.exists():
        return 0.0
    cost = float(c.metadata.get_cost(**kw))
    print(f"  {path.name:34s} {kw['schema']:11s} ${cost:,.4f}")
    if dry_run:
        return cost
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    c.timeseries.get_range(path=tmp, **kw)
    os.replace(tmp, path)
    return cost


def load_dbn(path: Path) -> pd.DataFrame:
    import databento as db

    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return db.DBNStore.from_file(path).to_df(map_symbols=False).reset_index()


def discover(defs: pd.DataFrame, root: str):
    """Options whose underlying is a <root> outright future, plus those futures."""
    fut_re = rf"^{re.escape(root)}[{MONTH_CODES}]\d{{1,2}}$"
    defs = defs.sort_values("ts_recv").drop_duplicates("instrument_id", keep="last")
    sym = defs["raw_symbol"].astype(str)
    futs = defs[(defs["instrument_class"] == "F") & sym.str.match(fut_re)]
    opts = defs[defs["instrument_class"].isin(["C", "P"])
                & defs["underlying"].astype(str).str.match(fut_re)]
    return opts, futs


def download(root, day, folder: Path, parents, dry_run, max_cost):
    c = client()
    s, e = window(day)
    total = 0.0
    defs_path = folder / "definition_ALL.dbn.zst"
    if parents is None or not defs_path.exists():
        print("Definitions (finds the option products on this root):")
        total += fetch(c, defs_path, dry_run, dataset=DATASET, schema="definition",
                       symbols="ALL_SYMBOLS", start=s, end=e)
        if dry_run and not defs_path.exists():
            print("  (dry run: option products not known yet, so statistics are not priced)")
            print(f"Estimated cost so far: ${total:,.4f}")
            sys.exit(0)
    opts, _ = discover(load_dbn(defs_path), root)
    if parents is None:
        parents = sorted(f"{a}.OPT" for a in opts["asset"].astype(str).str.strip().unique())
        if not parents:
            sys.exit(f"No options on {root} futures found on {day}")
        print(f"Option products on {root}: {' '.join(parents)}")
    print("Statistics (open interest + settlements):")
    reqs = [(folder / f"statistics_{p}.dbn.zst", p) for p in [*parents, f"{root}.FUT"]]
    todo_cost = sum(float(c.metadata.get_cost(dataset=DATASET, schema="statistics", symbols=[p],
                                              stype_in="parent", start=s, end=e))
                    for path, p in reqs if not path.exists())
    total += todo_cost
    print(f"Estimated cost: ${total:,.4f}")
    if dry_run:
        sys.exit(0)
    if total > max_cost:
        sys.exit(f"Estimated cost ${total:.2f} is above --max-cost ${max_cost:.2f}")
    for path, p in reqs:
        fetch(c, path, False, dataset=DATASET, schema="statistics", symbols=[p],
              stype_in="parent", start=s, end=e)
    return parents


# ====================================================================== chain
def last_stat(stats: pd.DataFrame, stat_type: int, col: str) -> pd.Series:
    s = stats[stats["stat_type"] == stat_type].copy()
    s[col] = pd.to_numeric(s[col], errors="coerce").astype(float)
    s = s[np.isfinite(s[col]) & (s[col].abs() < 1e15)]          # drop i64::MAX "undefined" sentinels
    s = s.sort_values("ts_recv").drop_duplicates("instrument_id", keep="last")
    return s.set_index("instrument_id")[col]


def build_chain(root, day, folder: Path, parents):
    opts, futs = discover(load_dbn(folder / "definition_ALL.dbn.zst"), root)
    assets = {p.removesuffix(".OPT") for p in parents}
    opts = opts[opts["asset"].astype(str).str.strip().isin(assets)]
    stats = pd.concat([load_dbn(folder / f"statistics_{p}.dbn.zst") for p in parents],
                      ignore_index=True)
    fstats = load_dbn(folder / f"statistics_{root}.FUT.dbn.zst")
    if stats.empty or fstats.empty:
        sys.exit("No statistics records (check the date: a holiday or weekend?)")

    fut_settle = last_stat(fstats, SETTLE_STAT, "price")
    fut_by_sym = futs.set_index("raw_symbol")["instrument_id"].map(fut_settle)
    chain = pd.DataFrame({
        "instrument_id": opts["instrument_id"].astype("int64").to_numpy(),
        "raw_symbol": opts["raw_symbol"].astype(str).to_numpy(),
        "asset": opts["asset"].astype(str).str.strip().to_numpy(),
        "underlying": opts["underlying"].astype(str).to_numpy(),
        "strike": pd.to_numeric(opts["strike_price"], errors="coerce").astype(float).to_numpy(),
        "is_call": (opts["instrument_class"] == "C").to_numpy(),
        "expiry": pd.to_datetime(opts["expiration"], utc=True).to_numpy(),
    })
    chain["expiry"] = pd.to_datetime(chain["expiry"], utc=True)
    chain["F"] = chain["underlying"].map(fut_by_sym).astype(float)
    if "underlying_id" in opts:
        by_id = pd.Series(pd.to_numeric(opts["underlying_id"], errors="coerce").to_numpy(),
                          index=chain.index).map(fut_settle)
        chain["F"] = chain["F"].fillna(by_id)
    chain["oi"] = chain["instrument_id"].map(last_stat(stats, OI_STAT, "quantity")).fillna(0).astype(float)
    chain["settle"] = chain["instrument_id"].map(last_stat(stats, SETTLE_STAT, "price")).astype(float)

    hh, mm = SETTLE_ET.get(root, (14, 30))
    now = pd.Timestamp(datetime(day.year, day.month, day.day, hh, mm, tzinfo=NY))
    chain["T"] = ((chain["expiry"] - now).dt.total_seconds() / (365.25 * 86400)).clip(lower=0)
    chain["dte"] = (chain["expiry"].dt.tz_convert(NY).dt.date - day).apply(lambda d: d.days)
    missing = chain["F"].isna() & (chain["oi"] > 0)
    if missing.any():
        print(f"Note: {missing.sum()} options with OI have no underlying settlement "
              f"(e.g. {chain.loc[missing, 'underlying'].head(3).tolist()}); left out")
    return chain[chain["F"].notna()].copy()


def add_iv(chain, r):
    """IV from OTM settlements; per-expiry smile fills the rest (x = ln(K/F)/sqrt(T))."""
    chain = chain.copy()
    chain["iv"], chain["iv_src"] = np.nan, ""
    for _, g in chain.groupby(["expiry", "underlying"]):
        T = float(g["T"].iloc[0])
        if T <= 0:
            continue
        df = math.exp(-r * T)
        F, K = g["F"].to_numpy(), g["strike"].to_numpy()
        is_call, px = g["is_call"].to_numpy(), g["settle"].to_numpy()
        otm = np.where(is_call, K >= F, K <= F)
        use = otm & np.isfinite(px) & (px > 0)
        iv = np.full(len(g), np.nan)
        if use.any():
            iv[use] = implied_vol(px[use], F[use], K[use], T, is_call[use], df)
        x = np.log(K / F) / math.sqrt(T)
        vega = df * F * norm_pdf(_d1(F, K, T, np.where(np.isfinite(iv), iv, 0.3))[0]) * math.sqrt(T)
        w = np.where(np.isfinite(iv), vega, 0.0)
        coef = fit_smile(x, iv, w)
        src = np.where(np.isfinite(iv), "settle", "")
        if coef is not None:
            lo, hi = x[w > 0].min(), x[w > 0].max()
            iv = np.where(np.isfinite(iv), iv, np.clip(np.polyval(coef, np.clip(x, lo, hi)), 0.03, 3.0))
            src = np.where(src == "", "smile", src)
        elif np.isfinite(iv).any():
            iv = np.where(np.isfinite(iv), iv, np.nanmedian(iv))
            src = np.where(src == "", "median", src)
        chain.loc[g.index, "iv"] = iv
        chain.loc[g.index, "iv_src"] = src
    return chain


def b76_gamma(F, K, T, vol, df):
    d1, sT = _d1(F, K, T, vol)
    return df * norm_pdf(d1) / (F * sT)


def contract_gex(c, r, mult, move=0.0):
    F = c["F"].to_numpy() * (1.0 + move)
    T = c["T"].to_numpy()
    g = b76_gamma(F, c["strike"].to_numpy(), T, c["iv"].to_numpy(), np.exp(-r * T))
    return np.where(c["is_call"], 1.0, -1.0) * g * c["oi"].to_numpy() * mult * F ** 2 * 0.01


def nearest_expiries(chain, n):
    """Keep the n soonest expiries that carry open interest."""
    d = chain["expiry"].dt.tz_convert(NY).dt.date
    keep = sorted(d[chain["oi"] > 0].unique())[:n]
    return chain[d.isin(keep)]


# ====================================================================== Yahoo ETF proxy
# Yahoo has no futures option chains, so --source yahoo uses the options on an ETF that tracks
# the same commodity and maps its strikes to futures prices by the ETF/futures price ratio.
PROXY_ETF = {"GC": "GLD", "MGC": "GLD", "CL": "USO", "MCL": "USO", "BZ": "BNO", "SI": "SLV",
             "NG": "UNG", "HG": "CPER", "ES": "SPY", "NQ": "QQQ", "RTY": "IWM", "YM": "DIA",
             "ZN": "IEF", "ZB": "TLT", "6E": "FXE"}


def yahoo_proxy(root, a):
    import equity_option_gex as eq

    etf = (a.proxy or PROXY_ETF.get(root, "")).upper()
    if not etf:
        sys.exit(f"No ETF proxy known for {root}: pass --proxy TICKER")
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("yfinance is not installed: pip install yfinance")
    chain, etf_px, _, day = eq.yahoo_chain(etf, None)
    fh = yf.Ticker(f"{root}=F").history(period="5d")
    if fh.empty:
        sys.exit(f"Yahoo has no price for {root}=F")
    fut_px = float(fh["Close"].iloc[-1])
    ratio = fut_px / etf_px
    if a.date and a.date != str(day):
        print(f"Note: Yahoo only has the latest session ({day}), not {a.date}")
    date_s = str(day)

    chain = chain[chain["dte"] >= a.min_dte]
    if a.max_dte is not None:
        chain = chain[chain["dte"] <= a.max_dte]
    if a.expiry:
        want = {datetime.strptime(e, "%Y-%m-%d").date() for e in a.expiry}
        chain = chain[chain["expiry"].dt.tz_convert(NY).dt.date.isin(want)]
    if a.nearest:
        chain = nearest_expiries(chain, a.nearest)
    chain = eq.add_iv(chain, etf_px, a.rate)
    tab, contracts = eq.gex_by_strike(chain, etf_px, a.rate)
    if tab.empty:
        sys.exit("No contracts with open interest and a usable IV after the filters")
    xs = np.linspace(-a.range, a.range, 301)
    prof = eq.gex_profile(contracts, etf_px, a.rate, xs)
    flip = zero_gamma(xs, prof, etf_px)

    etf_strikes = tab.index.to_numpy()
    tab.index = np.round(etf_strikes * ratio, 2)           # futures-equivalent strike
    tab["etf_strike"] = etf_strikes
    flip_f = flip * ratio if flip else None
    view = tab[(tab.index >= fut_px * (1 - a.range)) & (tab.index <= fut_px * (1 + a.range))]
    walls = ([(k, view.loc[k, "call_gex"], f"Call wall ({etf} {view.loc[k, 'etf_strike']:g}) ≈",
               wall_expiry(view, k, "call")) for k in view["call_gex"].nlargest(a.walls).index]
             + [(k, view.loc[k, "put_gex"], f"Put wall ({etf} {view.loc[k, 'etf_strike']:g}) ≈",
                 wall_expiry(view, k, "put")) for k in view["put_gex"].nsmallest(a.walls).index])

    out = Path(a.out) if a.out else Path(a.data) / root / date_s
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"gex_{root}_{date_s}_yahoo_{etf}"
    tab.to_csv(f"{stem}.csv", float_format="%.2f", index_label="futures_equiv_strike")
    exps = sorted(contracts["expiry"].dt.tz_convert(NY).dt.date.unique())
    exp_txt = (", ".join(map(str, exps)) if len(exps) <= 4
               else f"{len(exps)} expiries {exps[0]} to {exps[-1]}")
    title = (f"{root} gamma exposure from {etf} options (Yahoo proxy), close {date_s} · "
             f"{root}=F {fut_px:,.2f} · {etf} {etf_px:,.2f} · strikes × {ratio:.4f}\n"
             f"$ GEX is the {etf} options' · Expiries: {exp_txt}")
    plot_png(f"{stem}.png", title, view, fut_px, flip_f, walls, xs, prof, spot_label=f"{root}=F")
    plot_html(f"{stem}.html", title, view, fut_px, flip_f, walls, xs, prof, spot_label=f"{root}=F")

    net = tab["net_gex"].sum()
    print(f"\n{root} via {etf} options (Yahoo), {date_s}: {root}=F {fut_px:,.2f}, {etf} {etf_px:,.2f}, "
          f"ratio {ratio:.4f}")
    print(f"Used {len(contracts)} {etf} options with OI ({exp_txt}); "
          f"IV source {contracts['iv_src'].value_counts().to_dict()}")
    print(f"Total net GEX: {fmt_money(net)} per 1% move ({'dealers long gamma' if net > 0 else 'dealers short gamma'})")
    print(f"Zero-gamma flip: {flip_f:,.2f} ({flip:,.2f} in {etf})" if flip else "Zero-gamma flip: none within ±range")
    for k, v, kind, exp in walls:
        print(f"  {kind} {k:>9g}  {fmt_money(v):>8s}  ({(k / fut_px - 1) * 100:+.1f}%)"
              + (f"  mostly {exp}" if exp else ""))
    print(f"\nWrote {stem}.png, .html, .csv")
    if a.show:
        import webbrowser

        webbrowser.open(Path(f"{stem}.html").resolve().as_uri())


# ====================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="futures root: CL, GC, SI, NG, HG, ES, ...")
    ap.add_argument("date", nargs="?", help="trade date YYYY-MM-DD (not needed with --source yahoo)")
    ap.add_argument("--source", choices=["databento", "yahoo"], default="databento",
                    help="databento: real CME options (needs key, ~1 day delay without a live licence); "
                         "yahoo: free ETF-options proxy (GLD for GC, USO for CL, ...), latest session only")
    ap.add_argument("--proxy", help="ETF to use with --source yahoo (default from a table, e.g. GLD for GC)")
    ap.add_argument("--parents", nargs="+", help="option products, e.g. LO.OPT OG.OPT (default: all on the root)")
    ap.add_argument("--data", default="fut_opt_data", help="cache folder (default fut_opt_data)")
    ap.add_argument("--out", help="output folder (default: the day's data folder)")
    ap.add_argument("--multiplier", type=float, help="$ per 1.0 futures point (default from a table)")
    ap.add_argument("--rate", type=float, default=0.04, help="risk-free rate (default 0.04)")
    ap.add_argument("--min-dte", type=int, default=1, help="drop expiries closer than this (default 1)")
    ap.add_argument("--max-dte", type=int, default=None, help="only expiries up to this many days out")
    ap.add_argument("--nearest", type=int, metavar="N",
                    help="only the N nearest expiries that have open interest (after --min-dte)")
    ap.add_argument("--expiry", action="append", metavar="YYYY-MM-DD", help="only these expiries (repeatable)")
    ap.add_argument("--range", type=float, default=0.15, help="plot strikes within ±this fraction (default 0.15)")
    ap.add_argument("--walls", type=int, default=1, help="label the N largest call and put walls (default 1)")
    ap.add_argument("--max-cost", type=float, default=5.0, help="refuse downloads above this ($)")
    ap.add_argument("--dry-run", action="store_true", help="print the cost estimate and stop")
    ap.add_argument("--show", action="store_true", help="open the HTML chart in a browser")
    a = ap.parse_args()

    root = a.root.upper()
    if a.source == "yahoo":
        return yahoo_proxy(root, a)
    if not a.date:
        sys.exit("date is required with --source databento")
    day = datetime.strptime(a.date, "%Y-%m-%d").date()
    mult = a.multiplier or MULTIPLIERS.get(root)
    if not mult:
        sys.exit(f"No multiplier known for {root}: pass --multiplier ($ per 1.0 futures point)")
    folder = Path(a.data) / root / a.date
    parents = download(root, day, folder, a.parents, a.dry_run, a.max_cost)

    chain = build_chain(root, day, folder, parents)
    n_all, oi_all = len(chain), chain["oi"].sum()
    chain = chain[chain["dte"] >= a.min_dte]
    if a.max_dte is not None:
        chain = chain[chain["dte"] <= a.max_dte]
    if a.expiry:
        want = {datetime.strptime(e, "%Y-%m-%d").date() for e in a.expiry}
        chain = chain[chain["expiry"].dt.tz_convert(NY).dt.date.isin(want)]
    if a.nearest:
        chain = nearest_expiries(chain, a.nearest)
    chain = add_iv(chain[chain["oi"] > 0], a.rate)
    c = chain[np.isfinite(chain["iv"]) & (chain["T"] > 0)].copy()
    if c.empty:
        sys.exit("No options with open interest and a usable IV after the filters")
    c["gex"] = contract_gex(c, a.rate, mult)
    tab = aggregate_by_strike(c)

    # reference price: the future with the most option OI behind it
    ref_fut = c.groupby("underlying")["oi"].sum().idxmax()
    spot = float(c.loc[c["underlying"] == ref_fut, "F"].iloc[0])
    xs = np.linspace(-a.range, a.range, 301)
    prof = np.array([contract_gex(c, a.rate, mult, x).sum() for x in xs])
    flip = zero_gamma(xs, prof, spot)
    view = tab[(tab.index >= spot * (1 - a.range)) & (tab.index <= spot * (1 + a.range))]
    walls = ([(k, view.loc[k, "call_gex"], "Call wall", wall_expiry(view, k, "call"))
              for k in view["call_gex"].nlargest(a.walls).index]
             + [(k, view.loc[k, "put_gex"], "Put wall", wall_expiry(view, k, "put"))
                for k in view["put_gex"].nsmallest(a.walls).index])

    out = Path(a.out) if a.out else folder
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"gex_{root}_{a.date}"
    tab.to_csv(f"{stem}.csv", float_format="%.2f")
    exps = sorted(c["expiry"].dt.tz_convert(NY).dt.date.unique())
    dte_txt = f"DTE {a.min_dte}" + (f"-{a.max_dte}" if a.max_dte is not None else "+")
    exp_txt = (", ".join(map(str, exps)) if len(exps) <= 4
               else f"{len(exps)} expiries {exps[0]} to {exps[-1]}")
    title = (f"{root} options gamma exposure by strike, settlement {a.date} · {ref_fut} {spot:,.2f} · "
             f"OI as of prior day\nProducts: {' '.join(p.removesuffix('.OPT') for p in parents)} · "
             f"Expiries: {exp_txt}" + ("" if a.expiry else f" ({dte_txt})"))
    plot_png(f"{stem}.png", title, view, spot, flip, walls, xs, prof, spot_label=ref_fut)
    plot_html(f"{stem}.html", title, view, spot, flip, walls, xs, prof, spot_label=ref_fut)

    print(f"\n{root} {a.date}: reference future {ref_fut} settled {spot:,.2f}; multiplier ${mult:,.0f}/pt")
    print(f"Options: {n_all} listed, total OI {oi_all:,.0f}; used {len(c)} with OI ({exp_txt}); "
          f"IV source {c['iv_src'].value_counts().to_dict()}")
    net = tab["net_gex"].sum()
    print(f"Total net GEX: {fmt_money(net)} per 1% move ({'dealers long gamma' if net > 0 else 'dealers short gamma'})")
    print(f"Zero-gamma flip: {flip:,.2f}" if flip else "Zero-gamma flip: none within ±range")
    for k, v, kind, exp in walls:
        print(f"  {kind:9s} {k:>9g}  {fmt_money(v):>8s}  ({(k / spot - 1) * 100:+.1f}% from {ref_fut})"
              + (f"  mostly {exp}" if exp else ""))
    print(f"\nWrote {stem}.png, .html, .csv")
    if a.show:
        import webbrowser

        webbrowser.open(Path(f"{stem}.html").resolve().as_uri())


if __name__ == "__main__":
    main()
