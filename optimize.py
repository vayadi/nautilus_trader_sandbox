"""Parallel parameter optimization for EsVolRegimeGex (one backtest per process).

The grid can be Lean's optimization.json as-is:

  {"parameters": [{"name": "target-delta", "min": 0.10, "max": 0.20, "step": 0.05},
                  {"name": "ts-max", "min": 0.95, "max": 1.0, "step": 0.05}]}

or a plain dict of value lists:

  {"target-delta": [0.10, 0.16, 0.20], "ts-max": [0.95, 1.0], "gex-filter": [0, 1]}

  python optimize.py --catalog ./catalog --grid grid.json --start 2026-01-01 --end 2026-03-31 \
      --workers 8 --fixed contracts=5 --target sharpe

  --module es_gex_futures optimizes the standalone GEX futures strategy instead.

Every combination runs in its own process (the backtest itself is single-threaded), all
reading the same catalog. Each run's CSVs go to <out>/run_XXXX/, and the ranked table to
<out>/optimization.csv. Memory: roughly one backtest's footprint per worker, so lower
--workers or --chunk-size if RAM runs short.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import multiprocessing as mp
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

TARGETS = ("sharpe", "total_pnl", "calmar", "max_drawdown", "win_rate", "trades")


def frange(lo, hi, step):
    n = int(math.floor((hi - lo) / step + 1e-9)) + 1
    vals = [round(lo + i * step, 10) for i in range(n)]
    return [int(v) for v in vals] if all(float(x).is_integer() for x in (lo, step)) else vals


def load_grid(path):
    with open(path) as fh:
        g = json.load(fh)
    if isinstance(g, dict) and "parameters" in g:        # Lean optimization.json
        return {p["name"]: frange(float(p["min"]), float(p["max"]), float(p["step"]))
                if "step" in p else [p["value"]] for p in g["parameters"]}
    return {k: v if isinstance(v, list) else [v] for k, v in g.items()}


def metrics(out_dir, cash):
    m = dict(total_pnl=np.nan, sharpe=np.nan, max_drawdown=np.nan, calmar=np.nan,
             trades=0, win_rate=np.nan)
    sig = os.path.join(out_dir, "signals.csv")
    if os.path.exists(sig) and os.path.getsize(sig) > 1:
        s = pd.read_csv(sig)
        eq = pd.concat([pd.Series([cash]), s["equity"].astype(float)], ignore_index=True)
        r = eq.pct_change().dropna()
        m["total_pnl"] = float(eq.iloc[-1] - cash)
        if len(r) > 1 and r.std() > 0:
            m["sharpe"] = float(r.mean() / r.std() * math.sqrt(252))
        dd = (eq / eq.cummax() - 1).min()
        m["max_drawdown"] = float(dd)
        years = max(len(r) / 252, 1e-9)
        ann = (eq.iloc[-1] / cash) ** (1 / years) - 1
        m["calmar"] = float(ann / abs(dd)) if dd < 0 else np.nan
    tr = os.path.join(out_dir, "trades.csv")
    if os.path.exists(tr) and os.path.getsize(tr) > 1:
        t = pd.read_csv(tr)
        m["trades"] = len(t)
        pnl_col = "option_pnl" if "option_pnl" in t else "pnl"
        m["win_rate"] = float((t[pnl_col] > 0).mean()) if len(t) else np.nan
        if "kind" in t:
            for kind, cnt in t["kind"].value_counts().items():
                m[f"{kind}_trades"] = int(cnt)
    return m


def run_one(job):
    """Worker: one full backtest. Imports happen here so each process gets its own engine."""
    idx, combo, a = job
    out_dir = os.path.join(a["out"], f"run_{idx:04d}")
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    try:
        from nautilus_trader.backtest.node import BacktestNode
        import importlib
        mod = importlib.import_module(a["module"])
        build_config, parse_params = mod.build_config, mod.parse_params

        pairs = [f"{k}={v}" for k, v in {**a["fixed"], **combo}.items()]
        params, _ = parse_params(pairs, None)
        cfg = build_config(a["catalog"], a["start"], a["end"], a["cash"], a["fee"], params,
                           out_dir, venue=a["venue"], oi_path=a["oi"],
                           chunk_size=a["chunk_size"], log_level="ERROR")
        if isinstance(cfg, tuple):                     # es_gex_futures returns (cfg, info)
            cfg = cfg[0]
        node = BacktestNode(configs=[cfg])
        node.run()
        node.dispose()
        res = dict(run=idx, **combo, **metrics(out_dir, a["cash"]), seconds=round(time.time() - t0))
    except Exception:  # noqa: BLE001  (one bad combo must not kill the grid)
        with open(os.path.join(out_dir, "error.txt"), "w") as fh:
            fh.write(traceback.format_exc())
        res = dict(run=idx, **combo, error=traceback.format_exc().strip().splitlines()[-1],
                   seconds=round(time.time() - t0))
    with open(os.path.join(out_dir, "params.json"), "w") as fh:
        json.dump({**a["fixed"], **combo}, fh, indent=1)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--module", default="run_backtest",
                    help="run_backtest (EsVolRegimeGex) or es_gex_futures (GEX futures strategy)")
    ap.add_argument("--grid", required=True, help="Lean optimization.json or {name: [values]}")
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default="2026-03-31")
    ap.add_argument("--cash", type=float, default=1_000_000.0)
    ap.add_argument("--fee", type=float, default=2.0)
    ap.add_argument("--fixed", action="append", default=[], help="key=value held constant, repeatable")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--target", default="sharpe", choices=TARGETS)
    ap.add_argument("--venue", default="GLBX")
    ap.add_argument("--oi", default=None)
    ap.add_argument("--chunk-size", type=int, default=250_000,
                    help="records loaded per batch; lower = less memory (default 250,000)")
    ap.add_argument("--out", default="optimization")
    args = ap.parse_args()

    grid = load_grid(args.grid)
    names = list(grid)
    combos = [dict(zip(names, vals)) for vals in itertools.product(*grid.values())]
    fixed = dict(kv.split("=", 1) for kv in args.fixed)

    import importlib                                    # validate names before spawning
    importlib.import_module(args.module).parse_params([f"{k}={v}" for k, v in {**fixed, **combos[0]}.items()], None)

    os.makedirs(args.out, exist_ok=True)
    oi = args.oi
    if oi is None:
        cand = os.path.join(args.catalog, "open_interest.parquet")
        oi = cand if os.path.exists(cand) else ""
    shared = dict(module=args.module, catalog=os.path.abspath(args.catalog), start=args.start, end=args.end,
                  cash=args.cash, fee=args.fee, fixed=fixed, venue=args.venue,
                  oi=os.path.abspath(oi) if oi else "", chunk_size=args.chunk_size,
                  out=os.path.abspath(args.out))
    jobs = [(i, c, shared) for i, c in enumerate(combos)]
    workers = min(args.workers, len(jobs))
    print(f"{len(jobs)} combinations of {names} on {workers} worker processes")

    rows, t0 = [], time.time()
    ctx = mp.get_context("spawn")                      # fresh interpreter per worker
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futs = [pool.submit(run_one, j) for j in jobs]
        for n, f in enumerate(as_completed(futs), 1):
            r = f.result()
            rows.append(r)
            tag = r.get("error") or (f"sharpe={r['sharpe']:.2f} pnl={r['total_pnl']:,.0f} "
                                     f"dd={r['max_drawdown']:.1%} trades={r['trades']}")
            print(f"[{n}/{len(jobs)}] run_{r['run']:04d} {({k: r[k] for k in names})} "
                  f"{tag} ({r['seconds']}s)", flush=True)

    df = pd.DataFrame(rows)
    if args.target in df:   # higher is better for all targets (drawdown is stored as negative)
        df = df.sort_values(args.target, ascending=False, na_position="last")
    df.to_csv(os.path.join(args.out, "optimization.csv"), index=False)
    print(f"\nDone in {time.time() - t0:.0f}s. Top 10 by {args.target}:")
    cols = ["run"] + names + [c for c in ("sharpe", "total_pnl", "max_drawdown", "calmar",
                                          "trades", "win_rate", "error") if c in df]
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(df[cols].head(10).to_string(index=False))
    print(f"\nFull table: {os.path.join(os.path.abspath(args.out), 'optimization.csv')}")


if __name__ == "__main__":
    main()
