#!/usr/bin/env python3
"""
Gamma exposure (GEX) by strike for a US equity option chain (Databento OPRA) -> call/put walls.

  python equity_option_gex.py MU 2026-11-20                     # download (if needed) + plot
  python equity_option_gex.py MU 2026-11-20 --dry-run           # cost estimate only
  python equity_option_gex.py MU 2026-11-20 --max-dte 45 --range 0.2 --show
  python equity_option_gex.py MU 2026-11-20 --min-dte 0         # keep options expiring that day

Pipeline (one trading day, raw DBN files cached under eq_opt_data/<SYM>/<date>/, reruns are free):
  1. OPRA.PILLAR  definition   <SYM>.OPT (parent)  -> strike, expiry, call/put per contract
  2. OPRA.PILLAR  statistics   <SYM>.OPT           -> open interest (stat_type 9, published by
                                                       OCC before the open = OI at the prior close)
  3. OPRA.PILLAR  cbbo-1m      <SYM>.OPT, last --quote-minutes before the close -> bid/ask
  4. XNAS.ITCH    ohlcv-1m     <SYM>                -> spot at the close (or --spot, or put-call parity)

Then per expiry: implied vol from OTM mids, a quadratic smile fit fills strikes without a clean
quote (same method as cl_gex_futures.py), Black-Scholes gamma, and dollar GEX per 1% move:

    GEX = gamma * OI * 100 * S^2 * 0.01         dealers long calls (+), short puts (-)

Outputs next to the data (or --out): gex_<SYM>_<date>.png, .html (plotly, hover) and .csv
(per-strike call / put / net GEX), and prints spot, zero-gamma flip and the largest walls.

Requires: databento, numpy, pandas, matplotlib, plotly (conda env `nautilus`); env DATABENTO_API_KEY.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

NY = ZoneInfo("America/New_York")
OPT_DATASET = "OPRA.PILLAR"
EQ_DATASET = "XNAS.ITCH"
MULT = 100.0
OI_STAT = 9  # databento StatType.OPEN_INTEREST

# ====================================================================== pricing (as cl_gex_futures.py)
_SQRT2 = math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def norm_cdf(x):
    x = np.asarray(x, dtype=float)
    z = np.abs(x) / _SQRT2
    t = 1.0 / (1.0 + 0.3275911 * z)
    poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
            + 0.254829592) * t
    return 0.5 * (1.0 + np.sign(x) * (1.0 - poly * np.exp(-z * z)))


def norm_pdf(x):
    return _INV_SQRT_2PI * np.exp(-0.5 * np.asarray(x, dtype=float) ** 2)


def _d1(F, K, T, vol):
    sT = vol * np.sqrt(T)
    return (np.log(F / K) + 0.5 * sT * sT) / sT, sT


def black76(F, K, T, vol, is_call, df):
    d1, sT = _d1(F, K, T, vol)
    d2 = d1 - sT
    call = df * (F * norm_cdf(d1) - K * norm_cdf(d2))
    put = df * (K * norm_cdf(-d2) - F * norm_cdf(-d1))
    return np.where(is_call, call, put)


def implied_vol(price, F, K, T, is_call, df, iters=40):
    price, F, K, T = (np.asarray(a, dtype=float) for a in (price, F, K, T))
    is_call = np.asarray(is_call, dtype=bool)
    F, K, T, df, is_call = np.broadcast_arrays(F, K, T, np.asarray(df, dtype=float), is_call)
    intrinsic = df * np.where(is_call, np.maximum(F - K, 0.0), np.maximum(K - F, 0.0))
    upper = df * np.where(is_call, F, K)
    valid = (T > 0) & (price > intrinsic + 1e-10) & (price < upper)
    Ts = np.where(T > 0, T, 1.0)
    lo, hi = np.full(price.shape, 1e-4), np.full(price.shape, 5.0)
    vol = np.clip(np.sqrt(2.0 * np.pi / Ts) * price / (df * F), 0.02, 2.0)
    for _ in range(iters):
        diff = black76(F, K, Ts, vol, is_call, df) - price
        hi = np.where(diff > 0, vol, hi)
        lo = np.where(diff <= 0, vol, lo)
        d1, _ = _d1(F, K, Ts, vol)
        vega = df * F * norm_pdf(d1) * np.sqrt(Ts)
        newton = vol - diff / np.maximum(vega, 1e-12)
        vol = np.where((newton > lo) & (newton < hi), newton, 0.5 * (lo + hi))
    return np.where(valid, vol, np.nan)


def fit_smile(x, iv, w, rounds=2, k_mad=4.0):
    keep = np.isfinite(iv) & np.isfinite(x) & (w > 0)
    coef = None
    for _ in range(rounds):
        if keep.sum() < 4:
            return None
        coef = np.polyfit(x[keep], iv[keep], 2, w=np.sqrt(w[keep]))
        resid = iv - np.polyval(coef, x)
        mad = np.median(np.abs(resid[keep])) + 1e-6
        keep = keep & (np.abs(resid) <= k_mad * 1.4826 * mad)
    return coef


def bs_gamma(S, K, T, vol, r=0.0, q=0.0):
    """Black-Scholes spot gamma (via Black-76 on the forward)."""
    F = S * np.exp((r - q) * T)
    d1, sT = _d1(F, K, T, vol)
    return np.exp(-q * T) * norm_pdf(d1) / (S * sT)


# ====================================================================== Databento download
def client():
    import databento as db

    if not os.environ.get("DATABENTO_API_KEY"):
        sys.exit("DATABENTO_API_KEY is not set")
    return db.Historical()


def session_utc(day: date):
    o = datetime(day.year, day.month, day.day, 9, 30, tzinfo=NY).astimezone(timezone.utc)
    c = datetime(day.year, day.month, day.day, 16, 0, tzinfo=NY).astimezone(timezone.utc)
    return o, c


def requests_for(sym, day, quote_minutes):
    """[(file name, get_range kwargs)] for one day."""
    d0 = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    d1 = d0 + timedelta(days=1)
    _, close = session_utc(day)
    qstart = close - timedelta(minutes=quote_minutes)
    opt = dict(dataset=OPT_DATASET, symbols=[f"{sym}.OPT"], stype_in="parent")
    return [
        ("definition.dbn.zst", dict(opt, schema="definition", start=d0.isoformat(), end=close.isoformat())),
        ("statistics.dbn.zst", dict(opt, schema="statistics", start=d0.isoformat(), end=close.isoformat())),
        ("cbbo-1m.dbn.zst", dict(opt, schema="cbbo-1m", start=qstart.isoformat(), end=close.isoformat())),
        ("spot-ohlcv-1m.dbn.zst", dict(dataset=EQ_DATASET, symbols=[sym], stype_in="raw_symbol",
                                       schema="ohlcv-1m", start=(close - timedelta(minutes=30)).isoformat(),
                                       end=close.isoformat())),
    ]


def download(sym, day, folder: Path, quote_minutes, max_cost, dry_run, skip_spot):
    reqs = [(f, kw) for f, kw in requests_for(sym, day, quote_minutes)
            if not (skip_spot and f.startswith("spot"))]
    todo = [(f, kw) for f, kw in reqs if not (folder / f).exists()]
    if not todo:
        return
    c = client()
    total = 0.0
    for f, kw in todo:
        try:
            cost = float(c.metadata.get_cost(**kw))
        except Exception as e:  # noqa: BLE001
            print(f"  cost check failed for {f}: {e}")
            cost = float("nan")
        total += 0.0 if math.isnan(cost) else cost
        print(f"  {f:24s} {kw['dataset']:12s} {kw['schema']:11s} ${cost:,.4f}")
    print(f"Estimated cost: ${total:,.4f}")
    if dry_run:
        sys.exit(0)
    if total > max_cost:
        sys.exit(f"Estimated cost ${total:.2f} is above --max-cost ${max_cost:.2f}; rerun with a higher limit")
    folder.mkdir(parents=True, exist_ok=True)
    for f, kw in todo:
        path = folder / f
        tmp = path.with_suffix(path.suffix + ".part")
        try:
            c.timeseries.get_range(path=tmp, **kw)
        except Exception as e:  # noqa: BLE001
            if f.startswith("spot"):
                print(f"  spot download failed ({e}); falling back to put-call parity")
                continue
            raise
        os.replace(tmp, path)
        print(f"  saved {path}")


def load_dbn(path: Path, **kw) -> pd.DataFrame:
    import databento as db

    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return db.DBNStore.from_file(path).to_df(**kw)


# ====================================================================== chain assembly
def build_chain(folder: Path, day: date, sym: str, spot_override, quote_minutes):
    defs = load_dbn(folder / "definition.dbn.zst", map_symbols=False).reset_index()
    if defs.empty:
        sys.exit("No option definitions: check the symbol / date")
    defs = defs.sort_values("ts_recv").drop_duplicates("instrument_id", keep="last")
    defs = defs[defs["instrument_class"].isin(["C", "P"])]
    defs["underlying"] = defs["underlying"].astype(str).str.strip()
    defs = defs[defs["underlying"] == sym]  # drop adjusted / non-standard roots (e.g. MU1)
    chain = pd.DataFrame({
        "instrument_id": defs["instrument_id"].astype("int64"),
        "raw_symbol": defs["raw_symbol"].astype(str),
        "strike": pd.to_numeric(defs["strike_price"], errors="coerce").astype(float),
        "is_call": (defs["instrument_class"] == "C").to_numpy(),
        "expiry": pd.to_datetime(defs["expiration"], utc=True),
    })

    stats = load_dbn(folder / "statistics.dbn.zst", map_symbols=False).reset_index()
    if stats.empty:
        sys.exit("No statistics records: open interest unavailable")
    oi = stats[stats["stat_type"] == OI_STAT].sort_values("ts_recv")
    oi = oi.drop_duplicates("instrument_id", keep="last").set_index("instrument_id")["quantity"]
    chain["oi"] = chain["instrument_id"].map(oi).fillna(0).astype(float)

    q = load_dbn(folder / "cbbo-1m.dbn.zst", map_symbols=False).reset_index()
    if not q.empty:
        q = q.sort_values("ts_recv").drop_duplicates("instrument_id", keep="last").set_index("instrument_id")
        chain["bid"] = chain["instrument_id"].map(q["bid_px_00"]).astype(float)
        chain["ask"] = chain["instrument_id"].map(q["ask_px_00"]).astype(float)
    else:
        chain["bid"] = chain["ask"] = np.nan

    _, close = session_utc(day)
    chain["T"] = ((chain["expiry"] - pd.Timestamp(close)).dt.total_seconds() / (365.25 * 86400)).clip(lower=0)
    chain["dte"] = (chain["expiry"].dt.tz_convert(NY).dt.date - day).apply(lambda d: d.days)

    spot, src = spot_override, "--spot"
    if spot is None:
        bars = load_dbn(folder / "spot-ohlcv-1m.dbn.zst")
        if not bars.empty:
            spot, src = float(bars["close"].iloc[-1]), f"{EQ_DATASET} last 1m close"
    if spot is None:
        spot, src = parity_spot(chain), "put-call parity"
    return chain, float(spot), src


def yahoo_chain(sym, spot_override):
    """Latest session's full chain from Yahoo Finance (free, no history). -> chain, spot, src, day."""
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("yfinance is not installed: pip install yfinance")
    t = yf.Ticker(sym)
    hist = t.history(period="5d")
    if hist.empty:
        sys.exit(f"Yahoo returned no prices for {sym}")
    day = hist.index[-1].date()
    spot, src = (spot_override, "--spot") if spot_override else (float(hist["Close"].iloc[-1]), "Yahoo close")
    _, close = session_utc(day)
    frames = []
    for e in t.options:
        oc = t.option_chain(e)
        for df_, is_call in ((oc.calls, True), (oc.puts, False)):
            if df_.empty:
                continue
            frames.append(pd.DataFrame({
                "raw_symbol": df_["contractSymbol"],
                "strike": df_["strike"].astype(float),
                "is_call": is_call,
                "expiry": pd.Timestamp(datetime.strptime(e, "%Y-%m-%d").replace(hour=16, tzinfo=NY)),
                "oi": pd.to_numeric(df_["openInterest"], errors="coerce").fillna(0).astype(float),
                "bid": pd.to_numeric(df_["bid"], errors="coerce").astype(float),
                "ask": pd.to_numeric(df_["ask"], errors="coerce").astype(float),
                "yahoo_iv": pd.to_numeric(df_["impliedVolatility"], errors="coerce").astype(float),
            }))
    if not frames:
        sys.exit(f"Yahoo returned no option chain for {sym}")
    chain = pd.concat(frames, ignore_index=True)
    chain["expiry"] = pd.to_datetime(chain["expiry"]).dt.tz_convert("UTC")
    chain["T"] = ((chain["expiry"] - pd.Timestamp(close)).dt.total_seconds() / (365.25 * 86400)).clip(lower=0)
    chain["dte"] = (chain["expiry"].dt.tz_convert(NY).dt.date - day).apply(lambda d: d.days)
    return chain, spot, src, day


def parity_spot(chain):
    """Median of K + C - P over the nearest non-expiring expiry's liquid strikes (r~0)."""
    c = chain[(chain["dte"] >= 1) & (chain["bid"] > 0) & (chain["ask"] > 0)].copy()
    c["mid"] = 0.5 * (c["bid"] + c["ask"])
    exp = c["expiry"].min()
    c = c[c["expiry"] == exp]
    piv = c.pivot_table(index="strike", columns="is_call", values="mid")
    piv = piv.dropna()
    if piv.empty:
        sys.exit("Cannot infer spot; pass --spot")
    est = piv.index + piv[True] - piv[False]
    near = (piv[True] - piv[False]).abs().nsmallest(5).index
    return float(np.median(est.loc[near]))


def add_iv(chain, spot, r, max_rel_spread=0.35):
    """IV per contract: OTM mid where clean, else that expiry's smile fit, else its median."""
    chain = chain.copy()
    chain["iv"] = np.nan
    chain["iv_src"] = ""
    for exp, g in chain.groupby("expiry"):
        T = float(g["T"].iloc[0])
        if T <= 0:
            continue
        df = math.exp(-r * T)
        F = spot / df
        K, bid, ask = g["strike"].to_numpy(), g["bid"].to_numpy(), g["ask"].to_numpy()
        is_call = g["is_call"].to_numpy()
        mid = 0.5 * (bid + ask)
        quoted = (bid > 0) & (ask > bid) & ((ask - bid) <= max_rel_spread * mid)
        otm = np.where(is_call, K >= F, K <= F)
        use = quoted & otm
        iv = np.full(len(g), np.nan)
        if use.any():
            iv[use] = implied_vol(mid[use], F, K[use], T, is_call[use], df)
        x = np.log(K / F) / math.sqrt(T)
        vega = df * F * norm_pdf(_d1(F, K, T, np.where(np.isfinite(iv), iv, 0.5))[0]) * math.sqrt(T)
        w = np.where(use & np.isfinite(iv), vega / np.maximum(ask - bid, 1e-6), 0.0)
        coef = fit_smile(x, iv, w)
        src = np.where(np.isfinite(iv), "quote", "")
        if coef is not None:
            fit = np.clip(np.polyval(coef, np.clip(x, x[w > 0].min(), x[w > 0].max())), 0.05, 3.0)
            iv = np.where(np.isfinite(iv), iv, fit)
            src = np.where(src == "", "smile", src)
        elif "yahoo_iv" in g and (g["yahoo_iv"] > 0.05).any():          # no usable quotes (weekend)
            yiv = g["yahoo_iv"].to_numpy()
            ok = (yiv > 0.05) & (yiv < 3.0)
            fallback = np.median(yiv[ok])
            iv = np.where(np.isfinite(iv), iv, np.where(ok, yiv, fallback))
            src = np.where(src == "", "yahoo", src)
        elif np.isfinite(iv).any():
            iv = np.where(np.isfinite(iv), iv, np.nanmedian(iv))
            src = np.where(src == "", "median", src)
        chain.loc[g.index, "iv"] = iv
        chain.loc[g.index, "iv_src"] = src
    return chain


# ====================================================================== GEX
def gex_by_strike(chain, spot, r):
    c = chain[(chain["oi"] > 0) & np.isfinite(chain["iv"]) & (chain["T"] > 0)].copy()
    c["gamma"] = bs_gamma(spot, c["strike"].to_numpy(), c["T"].to_numpy(), c["iv"].to_numpy(), r)
    sign = np.where(c["is_call"], 1.0, -1.0)
    c["gex"] = sign * c["gamma"] * c["oi"] * MULT * spot ** 2 * 0.01
    return aggregate_by_strike(c), c


def aggregate_by_strike(c):
    """Contracts with strike, is_call, oi, gex, expiry -> per-strike call/put/net GEX table."""
    tab = c.pivot_table(index="strike", columns="is_call", values="gex", aggfunc="sum", fill_value=0.0)
    tab = tab.reindex(columns=[True, False], fill_value=0.0)
    tab.columns = ["call_gex", "put_gex"]
    oi = c.pivot_table(index="strike", columns="is_call", values="oi", aggfunc="sum", fill_value=0.0)
    oi = oi.reindex(columns=[True, False], fill_value=0.0)
    tab["call_oi"], tab["put_oi"] = oi[True], oi[False]
    tab["net_gex"] = tab["call_gex"] + tab["put_gex"]
    # which expiry contributes most of each strike's call / put gamma
    c["exp_date"] = c["expiry"].dt.tz_convert(NY).dt.date
    by = c.groupby(["strike", "is_call", "exp_date"])["gex"].sum().abs().reset_index()
    for is_call, side in ((True, "call"), (False, "put")):
        s = by[by["is_call"] == is_call]
        if s.empty:
            tab[f"{side}_top_exp"], tab[f"{side}_top_share"] = "", np.nan
            continue
        top = s.loc[s.groupby("strike")["gex"].idxmax()].set_index("strike")
        tot = s.groupby("strike")["gex"].sum()
        tab[f"{side}_top_exp"] = top["exp_date"].astype(str).reindex(tab.index).fillna("")
        tab[f"{side}_top_share"] = (top["gex"] / tot).reindex(tab.index)
    return tab.sort_index()


def wall_expiry(tab, k, side):
    e, sh = tab.loc[k, f"{side}_top_exp"], tab.loc[k, f"{side}_top_share"]
    return f"{e} ({sh:.0%})" if e and np.isfinite(sh) else ""


def gex_profile(contracts, spot, r, xs):
    """Total net GEX if spot moved to spot*(1+x), sticky-strike vols."""
    K, T, iv = (contracts[k].to_numpy()[None, :] for k in ("strike", "T", "iv"))
    w = (np.where(contracts["is_call"], 1.0, -1.0) * contracts["oi"]).to_numpy()[None, :]
    S = spot * (1.0 + np.asarray(xs))[:, None]
    g = bs_gamma(S, K, T, iv, r)
    return (g * w).sum(axis=1) * MULT * (S[:, 0] ** 2) * 0.01


def zero_gamma(xs, prof, spot):
    s = np.sign(prof)
    idx = np.flatnonzero(s[:-1] * s[1:] < 0)
    if len(idx) == 0:
        return None
    lvls = [spot * (1 + xs[j] - prof[j] * (xs[j + 1] - xs[j]) / (prof[j + 1] - prof[j])) for j in idx]
    return min(lvls, key=lambda v: abs(v - spot))


# ====================================================================== plotting
def fmt_money(v):
    a = abs(v)
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= div:
            return f"{v / div:,.1f}{suf}"
    return f"{v:,.0f}"


def plot_png(path, title, tab, spot, flip, walls, xs, prof, spot_label="Spot"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(13, 9), height_ratios=[3, 1.6])
    K = tab.index.to_numpy()
    width = np.median(np.diff(K)) * 0.8 if len(K) > 1 else 1.0
    ax.bar(K, tab["call_gex"] / 1e6, width, color="#2a9d8f", label="Call GEX (dealer long)")
    ax.bar(K, tab["put_gex"] / 1e6, width, color="#e76f51", label="Put GEX (dealer short)")
    ax.plot(K, tab["net_gex"] / 1e6, color="#264653", lw=1.6, marker="o", ms=3, label="Net GEX")
    ax.axhline(0, color="#888", lw=0.8)
    ax.axvline(spot, color="#1d3557", ls="--", lw=1.2, label=f"{spot_label} {spot:,.2f}")
    if flip:
        ax.axvline(flip, color="#9b5de5", ls=":", lw=1.5, label=f"Zero-gamma {flip:,.2f}")
    for k, v, kind, exp in walls:
        y = v / 1e6
        txt = f"{kind} {k:g}\n{fmt_money(v)}" + (f"\nmostly {exp}" if exp else "")
        ax.annotate(txt, (k, y), xytext=(0, 4 if y > 0 else -4),
                    textcoords="offset points", ha="center", va="bottom" if y > 0 else "top",
                    fontsize=8, fontweight="bold",
                    color="#2a9d8f" if kind.startswith("Call") else "#e76f51")
    ax.set_ylabel("GEX ($M per 1% move)")
    ax.margins(y=0.18)                               # room for the wall labels
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)

    ax2.plot(spot * (1 + xs), prof / 1e6, color="#264653", lw=1.6)
    ax2.fill_between(spot * (1 + xs), prof / 1e6, 0, where=prof >= 0, color="#2a9d8f", alpha=0.2)
    ax2.fill_between(spot * (1 + xs), prof / 1e6, 0, where=prof < 0, color="#e76f51", alpha=0.2)
    ax2.axhline(0, color="#888", lw=0.8)
    ax2.axvline(spot, color="#1d3557", ls="--", lw=1.2)
    if flip:
        ax2.axvline(flip, color="#9b5de5", ls=":", lw=1.5)
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xlabel("Strike / underlying price")
    ax2.set_ylabel("Total net GEX ($M)\nif spot moved here")
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_html(path, title, tab, spot, flip, walls, xs, prof, spot_label="Spot"):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.06,
                        subplot_titles=("GEX by strike", "Total net GEX vs underlying price"))
    K = tab.index
    hover = [f"Strike {k:g}<br>Call OI {co:,.0f} · Put OI {po:,.0f}<br>Call GEX {fmt_money(c)}"
             f" (mostly {wall_expiry(tab, k, 'call') or '-'})"
             f"<br>Put GEX {fmt_money(p)} (mostly {wall_expiry(tab, k, 'put') or '-'})<br>Net {fmt_money(n)}"
             for k, co, po, c, p, n in zip(K, tab["call_oi"], tab["put_oi"], tab["call_gex"],
                                           tab["put_gex"], tab["net_gex"])]
    fig.add_bar(x=K, y=tab["call_gex"], name="Call GEX", marker_color="#2a9d8f", hovertext=hover,
                hoverinfo="text", row=1, col=1)
    fig.add_bar(x=K, y=tab["put_gex"], name="Put GEX", marker_color="#e76f51", hovertext=hover,
                hoverinfo="text", row=1, col=1)
    fig.add_scatter(x=K, y=tab["net_gex"], name="Net GEX", mode="lines+markers",
                    line=dict(color="#264653"), hovertext=hover, hoverinfo="text", row=1, col=1)
    fig.add_scatter(x=spot * (1 + xs), y=prof, name="Net GEX profile", line=dict(color="#264653"),
                    showlegend=False, row=2, col=1)
    for row in (1, 2):
        fig.add_vline(x=spot, line_dash="dash", line_color="#1d3557", row=row, col=1,
                      annotation_text=f"{spot_label} {spot:,.2f}" if row == 1 else None)
        if flip:
            fig.add_vline(x=flip, line_dash="dot", line_color="#9b5de5", row=row, col=1,
                          annotation_text=f"Zero-gamma {flip:,.2f}" if row == 1 else None,
                          annotation_position="bottom right")
    for k, v, kind, exp in walls:
        fig.add_annotation(x=k, y=v, text=f"{kind} {k:g}" + (f"<br>mostly {exp}" if exp else ""),
                           showarrow=True, arrowhead=2, row=1, col=1)
    fig.update_layout(title=title, barmode="relative", template="plotly_white", height=850,
                      yaxis_title="$ per 1% move", yaxis2_title="$ per 1% move")
    fig.write_html(path, include_plotlyjs="cdn")


# ====================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("symbol", help="underlying ticker, e.g. MU")
    ap.add_argument("date", nargs="?", help="trading day YYYY-MM-DD (quotes at that day's close)")
    ap.add_argument("--source", choices=["databento", "yahoo"], default="databento",
                    help="databento (any past day, needs key; OPRA is licensed for ~1 day after) or "
                         "yahoo (free, latest session only, needs `pip install yfinance`)")
    ap.add_argument("--data", default="eq_opt_data", help="cache folder (default eq_opt_data)")
    ap.add_argument("--out", help="output folder (default: the day's data folder)")
    ap.add_argument("--spot", type=float, help="override the underlying price")
    ap.add_argument("--rate", type=float, default=0.04, help="risk-free rate (default 0.04)")
    ap.add_argument("--min-dte", type=int, default=1,
                    help="drop expiries closer than this (default 1: drops contracts expiring that day)")
    ap.add_argument("--max-dte", type=int, default=None, help="only expiries up to this many days out")
    ap.add_argument("--range", type=float, default=0.25, help="plot strikes within ±this fraction of spot")
    ap.add_argument("--walls", type=int, default=1,
                    help="label the N largest call and put walls (default 1: the call wall and the put wall)")
    ap.add_argument("--expiry", action="append", metavar="YYYY-MM-DD",
                    help="only these expiries (repeat the flag for several)")
    ap.add_argument("--quote-minutes", type=int, default=10, help="minutes of cbbo-1m before the close")
    ap.add_argument("--max-cost", type=float, default=5.0, help="refuse downloads above this ($)")
    ap.add_argument("--dry-run", action="store_true", help="print the cost estimate and stop")
    ap.add_argument("--no-spot-download", action="store_true", help="don't download the stock bars")
    ap.add_argument("--show", action="store_true", help="open the HTML chart in a browser")
    a = ap.parse_args()

    sym = a.symbol.upper()
    if a.source == "yahoo":
        chain, spot, src, day = yahoo_chain(sym, a.spot)
        if a.date and a.date != str(day):
            print(f"Note: Yahoo only has the latest session ({day}), not {a.date}")
        a.date = str(day)
        folder = Path(a.data) / sym / a.date
    else:
        if not a.date:
            sys.exit("date is required with --source databento")
        day = datetime.strptime(a.date, "%Y-%m-%d").date()
        folder = Path(a.data) / sym / a.date
        download(sym, day, folder, a.quote_minutes, a.max_cost, a.dry_run,
                 a.no_spot_download or a.spot is not None)
        chain, spot, src = build_chain(folder, day, sym, a.spot, a.quote_minutes)
    n_all, oi_all = len(chain), chain["oi"].sum()
    chain = chain[chain["dte"] >= a.min_dte]
    if a.max_dte is not None:
        chain = chain[chain["dte"] <= a.max_dte]
    if a.expiry:
        want = {datetime.strptime(e, "%Y-%m-%d").date() for e in a.expiry}
        chain = chain[chain["expiry"].dt.tz_convert(NY).dt.date.isin(want)]
        if chain.empty:
            sys.exit(f"No contracts for expiry {', '.join(a.expiry)} (after the DTE filter)")
    chain = add_iv(chain, spot, a.rate)
    tab, contracts = gex_by_strike(chain, spot, a.rate)
    if tab.empty:
        sys.exit("No contracts with open interest and a usable IV")

    xs = np.linspace(-a.range, a.range, 401)
    prof = gex_profile(contracts, spot, a.rate, xs)
    flip = zero_gamma(xs, prof, spot)
    view = tab[(tab.index >= spot * (1 - a.range)) & (tab.index <= spot * (1 + a.range))]
    walls = ([(k, view.loc[k, "call_gex"], "Call wall", wall_expiry(view, k, "call"))
              for k in view["call_gex"].nlargest(a.walls).index]
             + [(k, view.loc[k, "put_gex"], "Put wall", wall_expiry(view, k, "put"))
                for k in view["put_gex"].nsmallest(a.walls).index])

    out = Path(a.out) if a.out else folder
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"gex_{sym}_{a.date}"
    tab.to_csv(f"{stem}.csv", float_format="%.2f")
    exps = sorted(chain.loc[chain["oi"] > 0, "expiry"].dt.tz_convert(NY).dt.date.unique())
    dte_txt = f"DTE {a.min_dte}" + (f"-{a.max_dte}" if a.max_dte is not None else "+")
    exp_txt = (", ".join(map(str, exps)) if len(exps) <= 4
               else f"{len(exps)} expiries {exps[0]} to {exps[-1]}")
    title = (f"{sym} gamma exposure by strike, close {a.date} · spot {spot:,.2f} · OI as of prior close\n"
             f"Expiries: {exp_txt}" + ("" if a.expiry else f" ({dte_txt})"))
    plot_png(f"{stem}.png", title, view, spot, flip, walls, xs, prof)
    plot_html(f"{stem}.html", title, view, spot, flip, walls, xs, prof)

    srcs = contracts["iv_src"].value_counts().to_dict()
    print(f"\n{sym} {a.date}: spot {spot:,.2f} ({src})")
    print(f"Contracts: {n_all} listed, total OI {oi_all:,.0f}; used {len(contracts)} with OI "
          f"({dte_txt}, {len(exps)} expiries {exps[0] if exps else ''}..{exps[-1] if exps else ''}); IV source {srcs}")
    print(f"Total net GEX at spot: {fmt_money(tab['net_gex'].sum())} per 1% move "
          f"({'dealers long gamma' if tab['net_gex'].sum() > 0 else 'dealers short gamma'})")
    print(f"Zero-gamma flip: {flip:,.2f}" if flip else "Zero-gamma flip: none within ±range")
    for k, v, kind, exp in walls:
        print(f"  {kind:9s} {k:>8g}  {fmt_money(v):>8s}  ({(k / spot - 1) * 100:+.1f}% from spot)"
              + (f"  mostly {exp}" if exp else ""))
    print(f"\nWrote {stem}.png, .html, .csv")
    if a.show:
        import webbrowser

        webbrowser.open(Path(f"{stem}.html").resolve().as_uri())


if __name__ == "__main__":
    main()
