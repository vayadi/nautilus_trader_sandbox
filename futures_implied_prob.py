#!/usr/bin/env python3
"""
Option-implied probability distribution of a futures price at each expiry (risk-neutral).

  python futures_implied_prob.py CL 2026-11-20 --show                    # CME options via Databento
  python futures_implied_prob.py CL --source yahoo --show                # free proxy (USO options)
  python futures_implied_prob.py GC --source yahoo --nearest 2 --levels 3400 3600 3800
  python futures_implied_prob.py CL 2026-11-20 --parents LO.OPT --nearest 2

Method (Breeden-Litzenberger): per expiry, fit a smile iv(x), x = ln(K/F)/sqrt(T), to the
out-of-the-money options' implied vols, price calls on a fine strike grid with that smile
(Black-76) and take the second derivative:  density(K) = e^{rT} * d2C/dK2.
The smile is held flat beyond the last quoted strike, so far tails are approximate.

Data: the same as futures_option_gex.py. Databento = CME settlement prices (cached under
fut_opt_data/<ROOT>/<date>/, so a GEX run for that day makes this free). Yahoo = options on a
tracking ETF (GLD for gold, USO for crude, ...), with prices mapped to futures by the price ratio.

Outputs: implied_prob_<ROOT>_<date>[_yahoo_<ETF>].png / .html / .csv, and a table per expiry:
expected move, 5-95% and 16-84% ranges, P(above today's price), and P(below) each --levels price.
These are risk-neutral probabilities (what option prices imply), not forecasts.
"""
from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import equity_option_gex as eq
import futures_option_gex as fo

NY = fo.NY


# ====================================================================== smile -> density
_erf = np.vectorize(math.erf, otypes=[float])


def call_price(F, K, T, vol, df):
    """Black-76 call with an exact erf: the fast CDF approximation used for GEX has a tiny kink
    at the money that shows up as a notch once differentiated twice."""
    sT = vol * math.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sT * sT) / sT
    N = lambda z: 0.5 * (1.0 + _erf(z / math.sqrt(2.0)))          # noqa: E731
    return df * (F * N(d1) - K * N(d1 - sT))


def smile_vol(coef, x, lo, hi):
    """Quadratic smile inside [lo, hi], continued linearly (same slope) outside, so the implied
    density has no artificial spikes where the data ends (vol floored at 2%)."""
    d = np.polyder(coef)
    s_lo, s_hi = np.polyval(d, lo), np.polyval(d, hi)
    v = np.polyval(coef, np.clip(x, lo, hi))
    v = np.where(x < lo, np.polyval(coef, lo) + s_lo * (x - lo), v)
    v = np.where(x > hi, np.polyval(coef, hi) + s_hi * (x - hi), v)
    return np.clip(v, 0.02, 3.0)


def density_for_expiry(pts: pd.DataFrame, F: float, T: float, r: float, n=1500, width=5.0):
    """pts: strike, iv (OTM implied vols for one expiry). -> (K grid, density, cdf, atm_iv) or None."""
    pts = pts[np.isfinite(pts["iv"]) & (pts["iv"] > 0.01) & (pts["iv"] < 3.0)]
    if len(pts) < 5 or T <= 0:
        return None
    K, iv = pts["strike"].to_numpy(float), pts["iv"].to_numpy(float)
    x = np.log(K / F) / math.sqrt(T)
    w = np.exp(-0.5 * x ** 2 / 4.0)                   # trust near-the-money points most
    coef = eq.fit_smile(x, iv, w)
    if coef is None:
        return None
    atm = float(np.clip(np.polyval(coef, 0.0), 0.02, 3.0))
    sd = atm * math.sqrt(T)
    grid = np.linspace(F * math.exp(-width * sd), F * math.exp(width * sd), n)
    vol = smile_vol(coef, np.log(grid / F) / math.sqrt(T), x.min(), x.max())
    df = math.exp(-r * T)
    call = call_price(F, grid, T, vol, df)
    dens = np.gradient(np.gradient(call, grid), grid) / df
    dens = np.clip(np.nan_to_num(dens), 0.0, None)
    area = np.trapezoid(dens, grid)
    if area <= 0:
        return None
    dens /= area
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(grid))])
    cdf /= cdf[-1]
    return grid, dens, cdf, atm


def summarize(grid, dens, cdf, F, levels):
    q = lambda p: float(np.interp(p, cdf, grid))                    # noqa: E731
    out = {
        "mean": float(np.trapezoid(grid * dens, grid)),
        "exp_move": float(np.trapezoid(np.abs(grid - F) * dens, grid)),
        "p05": q(0.05), "p16": q(0.16), "p50": q(0.50), "p84": q(0.84), "p95": q(0.95),
        "p_above_now": 1.0 - float(np.interp(F, grid, cdf)),
    }
    for lv in levels or []:
        out[f"P(<{lv:g})"] = float(np.interp(lv, grid, cdf, left=0.0, right=1.0))
    return out


# ====================================================================== data
def databento_curves(root, a):
    if not a.date:
        sys.exit("date is required with --source databento")
    day = datetime.strptime(a.date, "%Y-%m-%d").date()
    folder = Path(a.data) / root / a.date
    parents = fo.download(root, day, folder, a.parents, a.dry_run, a.max_cost)
    chain = fo.build_chain(root, day, folder, parents)
    chain = fo.add_iv(chain[chain["dte"] >= a.min_dte], a.rate)
    pts = chain[chain["iv_src"] == "settle"]
    exps = []
    for exp, g in pts.groupby("expiry"):
        und = g.groupby("underlying")["oi"].sum().idxmax()        # main future for this expiry
        g = g[g["underlying"] == und]
        exps.append(dict(expiry=exp, pts=g, F=float(g["F"].iloc[0]), T=float(g["T"].iloc[0]),
                         label=und, oi=float(chain.loc[chain["expiry"] == exp, "oi"].sum())))
    ref = chain.groupby("underlying")["oi"].sum().idxmax()
    ref_px = float(chain.loc[chain["underlying"] == ref, "F"].iloc[0])
    meta = dict(ratio=1.0, ref=ref, ref_px=ref_px, date=a.date, tag="",
                src=f"CME {' '.join(p.removesuffix('.OPT') for p in parents)} settlements")
    return exps, meta


def yahoo_curves(root, a):
    import yfinance as yf

    etf = (a.proxy or fo.PROXY_ETF.get(root, "")).upper()
    if not etf:
        sys.exit(f"No ETF proxy known for {root}: pass --proxy TICKER")
    chain, etf_px, _, day = eq.yahoo_chain(etf, None)
    fh = yf.Ticker(f"{root}=F").history(period="5d")
    if fh.empty:
        sys.exit(f"Yahoo has no price for {root}=F")
    fut_px = float(fh["Close"].iloc[-1])
    chain = eq.add_iv(chain[chain["dte"] >= a.min_dte], etf_px, a.rate)
    otm = np.where(chain["is_call"], chain["strike"] >= etf_px, chain["strike"] <= etf_px)
    pts = chain[otm & chain["iv_src"].isin(["quote", "yahoo"])]
    exps = []
    for exp, g in pts.groupby("expiry"):
        T = float(g["T"].iloc[0])
        exps.append(dict(expiry=exp, pts=g, F=etf_px * math.exp(a.rate * T), T=T, label=etf,
                         oi=float(chain.loc[chain["expiry"] == exp, "oi"].sum())))
    meta = dict(ratio=fut_px / etf_px, ref=f"{root}=F", ref_px=fut_px, date=str(day),
                tag=f"_yahoo_{etf}", src=f"{etf} options (Yahoo), ×{fut_px / etf_px:.4f} to {root}")
    return exps, meta


# ====================================================================== plotting
COLORS = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#9b5de5", "#00b4d8", "#6a994e"]


def plot_png(path, title, curves, ref_px, ref):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(13, 9), height_ratios=[2, 1.3], sharex=True)
    for i, c in enumerate(curves):
        col = COLORS[i % len(COLORS)]
        s = c["stats"]
        lab = (f"{c['expiry']} ({c['dte']}d) · ±{s['exp_move']:,.2f} exp. move · "
               f"16-84%: {s['p16']:,.2f}-{s['p84']:,.2f}")
        ax.plot(c["grid"], c["dens"], color=col, lw=1.8, label=lab)
        ax.fill_between(c["grid"], c["dens"], 0, where=(c["grid"] >= s["p16"]) & (c["grid"] <= s["p84"]),
                        color=col, alpha=0.10)
        ax2.plot(c["grid"], 1 - c["cdf"], color=col, lw=1.6)
    for a_ in (ax, ax2):
        a_.axvline(ref_px, color="#1d3557", ls="--", lw=1.2)
        a_.grid(alpha=0.25)
    ax.text(ref_px, ax.get_ylim()[1] * 0.97, f" {ref} {ref_px:,.2f}", va="top", fontsize=8, color="#1d3557")
    ax.set_ylabel("Probability density")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax2.set_ylabel("P(price above X)")
    ax2.set_xlabel("Futures price at expiry")
    ax2.set_ylim(0, 1)
    ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    lo = min(c["stats"]["p05"] for c in curves)
    hi = max(c["stats"]["p95"] for c in curves)
    pad = 0.25 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_html(path, title, curves, ref_px, ref):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.06,
                        subplot_titles=("Implied probability density", "P(price above X)"))
    for i, c in enumerate(curves):
        col = COLORS[i % len(COLORS)]
        name = f"{c['expiry']} ({c['dte']}d)"
        fig.add_scatter(x=c["grid"], y=c["dens"], name=name, line=dict(color=col), legendgroup=name,
                        hovertemplate="%{x:,.2f}<br>density %{y:.4g}<extra>" + name + "</extra>", row=1, col=1)
        fig.add_scatter(x=c["grid"], y=1 - c["cdf"], name=name, line=dict(color=col), legendgroup=name,
                        showlegend=False, hovertemplate="P(above %{x:,.2f}) = %{y:.1%}<extra>" + name + "</extra>",
                        row=2, col=1)
    for row in (1, 2):
        fig.add_vline(x=ref_px, line_dash="dash", line_color="#1d3557", row=row, col=1,
                      annotation_text=f"{ref} {ref_px:,.2f}" if row == 1 else None)
    lo = min(c["stats"]["p05"] for c in curves)
    hi = max(c["stats"]["p95"] for c in curves)
    pad = 0.25 * (hi - lo)
    fig.update_xaxes(range=[lo - pad, hi + pad])
    fig.update_yaxes(tickformat=".0%", range=[0, 1], row=2, col=1)
    fig.update_layout(title=title.replace("\n", "<br>"), template="plotly_white", height=850, hovermode="x unified")
    fig.write_html(path, include_plotlyjs="cdn")


# ====================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="futures root: CL, GC, SI, NG, HG, ES, ...")
    ap.add_argument("date", nargs="?", help="trade date YYYY-MM-DD (not needed with --source yahoo)")
    ap.add_argument("--source", choices=["databento", "yahoo"], default="databento")
    ap.add_argument("--proxy", help="ETF for --source yahoo (default: GLD for GC, USO for CL, ...)")
    ap.add_argument("--parents", nargs="+", help="CME option products, e.g. LO.OPT (default: all on the root)")
    ap.add_argument("--nearest", type=int, default=3, help="the N nearest expiries (default 3)")
    ap.add_argument("--expiry", action="append", metavar="YYYY-MM-DD", help="only these expiries (repeatable)")
    ap.add_argument("--min-dte", type=int, default=1, help="skip expiries closer than this (default 1)")
    ap.add_argument("--min-oi", type=float, default=100, help="skip expiries with less total OI (default 100)")
    ap.add_argument("--levels", type=float, nargs="+", help="futures prices to report P(below) for")
    ap.add_argument("--rate", type=float, default=0.04)
    ap.add_argument("--data", default="fut_opt_data")
    ap.add_argument("--out", help="output folder (default: the day's data folder)")
    ap.add_argument("--max-cost", type=float, default=5.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--show", action="store_true", help="open the HTML chart in a browser")
    a = ap.parse_args()

    root = a.root.upper()
    exps, m = yahoo_curves(root, a) if a.source == "yahoo" else databento_curves(root, a)
    exps = sorted((e for e in exps if e["oi"] >= a.min_oi), key=lambda e: e["expiry"])
    if a.expiry:
        want = {datetime.strptime(e, "%Y-%m-%d").date() for e in a.expiry}
        exps = [e for e in exps if e["expiry"].tz_convert(NY).date() in want]

    day = datetime.strptime(m["date"], "%Y-%m-%d").date()
    curves = []
    for e in exps:
        res = density_for_expiry(e["pts"], e["F"], e["T"], a.rate)
        if res is None:
            continue
        grid, dens, cdf, atm = res
        grid_f = grid * m["ratio"]                                 # to futures prices (1.0 for CME)
        dens_f = dens / m["ratio"]
        F_f = e["F"] * m["ratio"]
        exp_d = e["expiry"].tz_convert(NY).date()
        curves.append(dict(expiry=exp_d, dte=(exp_d - day).days, grid=grid_f, dens=dens_f, cdf=cdf,
                           atm=atm, F=F_f, label=e["label"],
                           stats=summarize(grid_f, dens_f, cdf, F_f, a.levels)))
        if not a.expiry and len(curves) >= a.nearest:
            break
    if not curves:
        sys.exit("No expiry had enough options to fit a smile")

    out = Path(a.out) if a.out else Path(a.data) / root / m["date"]
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"implied_prob_{root}_{m['date']}{m['tag']}"
    title = (f"{root} option-implied (risk-neutral) price distribution, {m['date']} close · "
             f"{m['ref']} {m['ref_px']:,.2f}\nSource: {m['src']} · shaded = 16-84% range")
    plot_png(f"{stem}.png", title, curves, m["ref_px"], m["ref"])
    plot_html(f"{stem}.html", title, curves, m["ref_px"], m["ref"])
    pd.concat([pd.DataFrame({"expiry": c["expiry"], "price": c["grid"], "density": c["dens"],
                             "p_below": c["cdf"]}) for c in curves]).to_csv(f"{stem}.csv", index=False)

    rows = []
    for c in curves:
        s = c["stats"]
        row = {"expiry": c["expiry"], "dte": c["dte"], "under": c["label"], "fwd": round(c["F"], 2),
               "ATM IV": f"{c['atm']:.1%}", "exp. move": f"±{s['exp_move']:,.2f}",
               "5-95%": f"{s['p05']:,.2f}-{s['p95']:,.2f}", "16-84%": f"{s['p16']:,.2f}-{s['p84']:,.2f}",
               "P(above fwd)": f"{s['p_above_now']:.0%}"}
        for lv in a.levels or []:
            row[f"P(<{lv:g})"] = f"{s[f'P(<{lv:g})']:.0%}"
        rows.append(row)
    print(f"\n{title}\n")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nWrote {stem}.png, .html, .csv")
    if a.show:
        import webbrowser

        webbrowser.open(Path(f"{stem}.html").resolve().as_uri())


if __name__ == "__main__":
    main()
