"""Equity curve of one or more backtest / optimizer runs (any strategy that writes signals.csv).

  python plot_equity.py opt_gamma_short_full --run 26              # one optimizer run
  python plot_equity.py opt_gamma_short_full --top 5               # best 5 by --target (default sharpe)
  python plot_equity.py opt_gamma_short_full --run 26 --run 24 results_gamma   # mix runs and result folders
  python plot_equity.py results_gamma --csv                        # also write equity_curve.csv
  python plot_equity.py opt_gamma_short_full --run 26 --format html --show   # plotly only, open browser

For each run it reads signals.csv (daily equity) and, if present, trades.csv (marks each
cycle's close, green = win, red = loss). Saves a PNG (matplotlib) and an interactive HTML
(plotly: zoom, hover shows each trade's dates, reason and P&L split) next to the first run, or
to --out, and prints a short summary. --format png | html | both (default both). With --csv it also writes each run's
date/equity/drawdown/daily P&L to <run>/equity_curve.csv.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd


def resolve(paths, runs, top, target):
    """-> list of (label, run_dir)."""
    out = []
    for p in paths:
        opt_csv = os.path.join(p, "optimization.csv")
        if os.path.exists(opt_csv) and (runs or top):
            tab = pd.read_csv(opt_csv)
            sel = [int(r) for r in runs or []]
            if top:
                sel += tab.sort_values(target, ascending=False)["run"].head(top).astype(int).tolist()
            for r in dict.fromkeys(sel):
                d = os.path.join(p, f"run_{r:04d}")
                params = {}
                pj = os.path.join(d, "params.json")
                if os.path.exists(pj):
                    params = json.load(open(pj))
                    params = {k: v for k, v in params.items() if k not in ("oi-source", "side")}
                label = f"{os.path.basename(os.path.normpath(p))} run {r}"
                if params:
                    label += " (" + ", ".join(f"{k}={v}" for k, v in params.items()) + ")"
                out.append((label, d))
        elif os.path.exists(os.path.join(p, "signals.csv")):
            out.append((os.path.basename(os.path.normpath(p)), p))
        elif os.path.exists(opt_csv):
            raise SystemExit(f"{p} is an optimizer folder: pass --run N or --top K")
        else:
            raise SystemExit(f"No signals.csv in {p}")
    return out


def load(run_dir, cash):
    s = pd.read_csv(os.path.join(run_dir, "signals.csv"), parse_dates=["date"])
    eq = s.set_index("date")["equity"].astype(float)
    start = cash
    df = pd.DataFrame({"equity": eq})
    df["pnl"] = df["equity"] - start
    df["daily_pnl"] = df["equity"].diff().fillna(df["equity"].iloc[0] - start)
    df["drawdown"] = df["equity"] - df["equity"].cummax().clip(lower=start)
    tr = os.path.join(run_dir, "trades.csv")
    trades = pd.read_csv(tr, parse_dates=["opened", "closed"]) if os.path.exists(tr) and os.path.getsize(tr) > 1 else None
    return df, trades, start


def trade_points(df, trades):
    """Closed trades placed on the curve: (day, y, row) per trade."""
    pts = []
    for _, t in trades.iterrows():
        day = pd.Timestamp(t["closed"]).normalize()
        y = df["pnl"].asof(day) if day >= df.index[0] else 0.0
        pts.append((day, y, t))
    return pts


def plot_png(series, out, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    for label, df, trades in series:
        line, = ax1.plot(df.index, df["pnl"], lw=1.6, label=label)
        ax2.fill_between(df.index, df["drawdown"], 0, color=line.get_color(), alpha=0.25, step="post")
        if trades is not None and len(series) == 1:
            for day, y, t in trade_points(df, trades):
                ax1.scatter(day, y, s=28, zorder=3, color="tab:green" if t["pnl"] > 0 else "tab:red")
    ax1.axhline(0, color="grey", lw=0.8)
    ax1.set_ylabel("P&L (USD)")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(alpha=0.3)
    ax2.set_ylabel("drawdown (USD)")
    ax2.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"png  -> {out}")
    if show:
        plt.show()


def plot_html(series, out, show):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("plotly is not installed: pip install plotly   (skipping the HTML plot)")
        return
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28],
                        vertical_spacing=0.04, subplot_titles=("P&L (USD)", "Drawdown (USD)"))
    for label, df, trades in series:
        fig.add_trace(go.Scatter(x=df.index, y=df["pnl"], mode="lines", name=label, legendgroup=label,
                                 customdata=np.c_[df["equity"], df["daily_pnl"]],
                                 hovertemplate="%{x|%Y-%m-%d}<br>P&L %{y:,.0f}<br>equity %{customdata[0]:,.0f}"
                                               "<br>day %{customdata[1]:+,.0f}<extra></extra>"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["drawdown"], mode="lines", fill="tozeroy", line_shape="hv",
                                 name=f"{label} drawdown", legendgroup=label, showlegend=False,
                                 hovertemplate="%{x|%Y-%m-%d}<br>drawdown %{y:,.0f}<extra></extra>"),
                      row=2, col=1)
        if trades is not None:
            pts = trade_points(df, trades)
            def col(name, t):
                return t[name] if name in t and pd.notna(t[name]) else np.nan
            text = []
            for _, _, t in pts:
                lines = [f"{t.get('kind', 'trade')} {pd.Timestamp(t['opened']):%m-%d %H:%M} -> "
                         f"{pd.Timestamp(t['closed']):%m-%d %H:%M}", f"exit: {t.get('reason', '')}",
                         f"P&L {t['pnl']:,.0f}"]
                if "opt_pnl" in t:
                    lines.append(f"options {col('opt_pnl', t):,.0f} / hedges {col('hedge_pnl', t):,.0f} "
                                 f"/ fees {-col('fees', t):,.0f}")
                if "iv_entry" in t and "rv_hold" in t:
                    lines.append(f"IV {col('iv_entry', t):.1%} vs RV held {col('rv_hold', t):.1%}")
                if "put" in t and "call" in t:
                    lines.append(f"{t['put']} / {t['call']}")
                text.append("<br>".join(lines))
            fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="markers",
                                     name=f"{label} trades", legendgroup=label, showlegend=False,
                                     marker=dict(size=9, line=dict(width=1, color="white"),
                                                 color=["#2ca02c" if p[2]["pnl"] > 0 else "#d62728" for p in pts]),
                                     text=text, hovertemplate="%{text}<extra></extra>"),
                          row=1, col=1)
    fig.add_hline(y=0, line_color="grey", line_width=1, row=1, col=1)
    fig.update_layout(template="plotly_white", height=750, hovermode="closest",
                      legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0, font=dict(size=10)),
                      margin=dict(l=60, r=20, t=80, b=40))
    fig.update_yaxes(tickformat=",.0f")
    fig.update_xaxes(rangeslider=dict(visible=False))
    fig.write_html(out, include_plotlyjs=True)
    print(f"html -> {out}")
    if show:
        fig.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="optimizer folders (with --run/--top) or result folders")
    ap.add_argument("--run", action="append", help="run number in an optimizer folder, repeatable")
    ap.add_argument("--top", type=int, default=0, help="also plot the best K runs")
    ap.add_argument("--target", default="sharpe", help="column of optimization.csv for --top")
    ap.add_argument("--cash", type=float, default=1_000_000.0, help="starting cash of the backtest")
    ap.add_argument("--format", default="both", choices=["png", "html", "both"])
    ap.add_argument("--out", default=None,
                    help="output path without extension (default <first run>/equity_curve)")
    ap.add_argument("--csv", action="store_true", help="also write <run>/equity_curve.csv")
    ap.add_argument("--show", action="store_true", help="open the plot(s) as well")
    a = ap.parse_args()

    series = []
    for label, d in resolve(a.paths, a.run, a.top, a.target):
        df, trades, start = load(d, a.cash)
        series.append((label, df, trades))
        dd = df["drawdown"].min()
        n = 0 if trades is None else len(trades)
        win = "" if not n else f", win rate {(trades['pnl'] > 0).mean():.0%}"
        r = df["equity"].pct_change().dropna()
        sharpe = r.mean() / r.std() * np.sqrt(252) if len(r) > 1 and r.std() > 0 else float("nan")
        print(f"{label}\n  {df.index[0]:%Y-%m-%d} .. {df.index[-1]:%Y-%m-%d}: P&L {df['pnl'].iloc[-1]:,.0f}, "
              f"max drawdown {dd:,.0f}, Sharpe {sharpe:.2f}, {n} trades{win}")
        if a.csv:
            p = os.path.join(d, "equity_curve.csv")
            df.to_csv(p, index_label="date")
            print(f"  -> {p}")
        first = first if series[1:] else d
    base = a.out or os.path.join(first, "equity_curve")
    base = os.path.splitext(base)[0] if base.endswith((".png", ".html")) else base
    if a.format in ("png", "both"):
        plot_png(series, base + ".png", a.show)
    if a.format in ("html", "both"):
        plot_html(series, base + ".html", a.show)


if __name__ == "__main__":
    main()
