"""VAR model for HUF interest rates driven by EUR, USD, JPY, GBP, CHF, AUD rates.

Estimation, lag selection, impulse/MA representation and forecast intervals use statsmodels' VAR
(statsmodels.tsa.api.VAR, OLS per equation on numpy/LAPACK). Data handling uses polars. The
parameter search runs the walk-forward backtests of each parameter set in parallel threads
(--executor process is available too; statsmodels' Python overhead holds the GIL, so processes
scale better on many cores).

Input: the panel written by rates_VectorAutoRegression_data.py (date + one column per currency, percent).

Commands
  fit       stationarity tests (ADF/KPSS), lag-order table (AIC/BIC/HQ/FPE), fitted VAR, stability,
            residual whiteness, Granger causality of every currency -> HUF.
  forecast  h-step forecast of every currency with (1-alpha) bands from the VAR MSE matrices;
            in 'diff' mode the bands are for the cumulated level path. Writes CSV + fan chart PNG
            (--bands, default 50/80/95%; --h 126 = half a year).
  optimize  walk-forward grid search (lags x transform x window x frequency x input set),
            scored on out-of-sample target RMSE at --horizon vs a random walk, in parallel.

--target picks the modelled currency (default HUF; any panel column, or ALL to loop over every one).
Outputs for a target other than HUF get a _<CCY> suffix (best_params_EUR.json, forecast_EUR.csv ...).

  python rates_VectorAutoRegression.py fit      --data rates_VectorAutoRegression/data/rates_official_10y.parquet
  python rates_VectorAutoRegression.py optimize --data rates_VectorAutoRegression/data/rates_official_10y.parquet --grid rates_VectorAutoRegression/rates_VectorAutoRegression_grid.json \
         --horizon 20 --test-years 2 --workers 8
  python rates_VectorAutoRegression.py forecast --data rates_VectorAutoRegression/data/rates_official_10y.parquet --params rates_VectorAutoRegression/results/best_params.json --h 126
  python rates_VectorAutoRegression.py optimize --data ... --grid rates_VectorAutoRegression/rates_VectorAutoRegression_grid.json --target ALL --executor process
  python rates_VectorAutoRegression.py forecast --data ... --target ALL --h 126      # uses best_params[_CCY].json per currency
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import polars as pl

warnings.filterwarnings("ignore")
TARGET = "HUF"  # default target
BDAYS = {"D": 1, "W": 5}


# ------------------------------------------------------------------------------------------ data
def load_panel(path: str, target: str = TARGET) -> pl.DataFrame:
    df = pl.read_parquet(path) if path.endswith(".parquet") else pl.read_csv(path, try_parse_dates=True)
    df = df.with_columns(pl.col("date").cast(pl.Date)).sort("date")
    if target != "ALL" and target not in df.columns:
        sys.exit(f"{path} has no {target} column")
    return df


def ccys_of(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c != "date"]


def targets_of(a, df: pl.DataFrame) -> list[str]:
    t = a.target.upper()
    return ccys_of(df) if t == "ALL" else [t]


def suffix(target: str) -> str:
    """File-name suffix: none for HUF (backwards compatible), _<CCY> otherwise."""
    return "" if target == TARGET else f"_{target}"


def resample(df: pl.DataFrame, freq: str) -> pl.DataFrame:
    if freq == "D":
        return df
    return df.group_by_dynamic("date", every="1w", label="right").agg(pl.all().exclude("date").last()).drop_nulls()


@dataclass(frozen=True)
class Params:
    lags: int | str = 2           # int, or "aic" / "bic" / "hqic" / "fpe" (selected per fit, max --maxlags)
    transform: str = "diff"       # "diff" (VAR on daily changes) or "level"
    window: int = 0               # rolling estimation window in observations, 0 = expanding
    freq: str = "D"               # "D" daily, "W" weekly (last obs of the week)
    inputs: tuple = ("EUR", "USD", "JPY", "GBP", "CHF", "AUD")
    target: str = TARGET

    @property
    def cols(self) -> list[str]:
        return [self.target] + [c for c in self.inputs if c != self.target]

    def label(self) -> str:
        ins = [c for c in self.inputs if c != self.target]
        return f"p={self.lags} {self.transform} win={self.window or 'exp'} {self.freq} in={'+'.join(ins) or '-'}"


def design(df: pl.DataFrame, p: Params) -> tuple[np.ndarray, np.ndarray, list]:
    """Returns (levels Y, model data X, dates) for the chosen columns/frequency/transform.
    X[t] corresponds to Y[t] (diff mode drops the first row of Y/dates)."""
    d = resample(df.select(["date"] + p.cols), p.freq)
    y = d.select(p.cols).to_numpy().astype(float)
    dates = d["date"].to_list()
    if p.transform == "diff":
        return y[1:], np.diff(y, axis=0), dates[1:]
    return y, y.copy(), dates


# ----------------------------------------------------------------------------------------- model
def fit_var(x: np.ndarray, lags, maxlags: int = 15):
    from statsmodels.tsa.api import VAR
    m = VAR(x)
    if isinstance(lags, str):
        return m.fit(maxlags=maxlags, ic=lags, trend="c")
    return m.fit(int(lags), trend="c")


def forecast_path(res, x: np.ndarray, y_last: np.ndarray, h: int, transform: str, alpha: float | None):
    """Point forecast of levels for steps 1..h and (optionally) +/- z*sd bands.
    diff: levels = last level + cumsum of forecast changes; Var of the cumulated sum uses the
    cumulated MA matrices C_j = sum_{i<=j} Psi_i:  sum_{j<h} C_j Sigma_u C_j'."""
    p = max(res.k_ar, 1)
    fc = res.forecast(x[-p:], steps=h) if res.k_ar > 0 else np.repeat(res.params[:1], h, axis=0)
    if transform == "diff":
        mean = y_last + np.cumsum(fc, axis=0)
    else:
        mean = fc
    if alpha is None:
        return mean, None
    from scipy.stats import norm
    psi = res.ma_rep(h - 1)                     # (h, k, k)
    sig = np.asarray(res.sigma_u)
    if transform == "diff":
        c = np.cumsum(psi, axis=0)
        var = np.cumsum(np.einsum("hij,jk,hlk->hil", c, sig, c), axis=0)
    else:
        var = np.cumsum(np.einsum("hij,jk,hlk->hil", psi, sig, psi), axis=0)
    sd = np.sqrt(np.maximum(np.diagonal(var, axis1=1, axis2=2), 0))
    return mean, norm.ppf(1 - alpha / 2) * sd


# ------------------------------------------------------------------------------------------- fit
def cmd_fit(a):
    df = load_panel(a.data, a.target.upper())
    for t in targets_of(a, df):
        if len(targets_of(a, df)) > 1:
            print(f"\n{'=' * 30} {t} {'=' * 30}")
        fit_one(a, df, params_from_args(a, t, ccys_of(df)))


def fit_one(a, df, p: Params):
    from statsmodels.tsa.api import VAR
    from statsmodels.tsa.stattools import adfuller, kpss
    T = p.target
    y, x, dates = design(df, p)
    print(f"{len(dates)} obs {dates[0]} .. {dates[-1]}   {p.label()}\n")
    print("Stationarity (ADF p<0.05 = stationary; KPSS p<0.05 = NOT stationary)")
    print(f"  {'':<4}{'ADF lvl':>9}{'KPSS lvl':>9}{'ADF diff':>9}{'KPSS diff':>10}")
    lv = resample(df.select(["date"] + p.cols), p.freq).select(p.cols).to_numpy()
    for i, c in enumerate(p.cols):
        s, ds = lv[:, i], np.diff(lv[:, i])
        with warnings.catch_warnings():  # statsmodels re-enables its KPSS InterpolationWarning on import
            warnings.simplefilter("ignore")
            r = [adfuller(s, autolag="AIC")[1], kpss(s, nlags="auto")[1], adfuller(ds, autolag="AIC")[1], kpss(ds, nlags="auto")[1]]
        print(f"  {c:<4}" + "".join(f"{v:9.3f}" for v in r[:3]) + f"{r[3]:10.3f}")

    sel = VAR(x).select_order(maxlags=a.maxlags, trend="c")
    print(f"\nLag order selection (max {a.maxlags}): { {k: int(v) for k, v in sel.selected_orders.items()} }")
    lags = p.lags if not isinstance(p.lags, str) else sel.selected_orders[p.lags]
    res = VAR(x).fit(max(int(lags), 1), trend="c")
    print(f"Using p={res.k_ar}  stable={res.is_stable()}  AIC={res.aic:.3f} BIC={res.bic:.3f}")
    try:
        wt = res.test_whiteness(nlags=max(res.k_ar + 5, 10))
        print(f"Residual whiteness (Portmanteau, {wt.df} df): stat={wt.test_statistic:.1f} p={wt.pvalue:.3f}")
    except Exception as e:  # noqa: BLE001
        print(f"whiteness test skipped: {e}")

    print(f"\nGranger causality -> {T} (F-test)")
    for c in p.cols[1:]:
        g = res.test_causality(0, [p.cols.index(c)], kind="f")
        print(f"  {c:<4} F={g.test_statistic:7.2f}  p={g.pvalue:.4f}{'  *' if g.pvalue < 0.05 else ''}")
    g = res.test_causality(0, list(range(1, len(p.cols))), kind="f")
    print(f"  all  F={g.test_statistic:7.2f}  p={g.pvalue:.4f}")

    print(f"\n{T} equation (coef, t-stat)")
    names = res.exog_names
    for j, nm in enumerate(names):
        import re
        nm = re.sub(r"y(\d+)", lambda m: p.cols[int(m.group(1)) - 1], nm)
        print(f"  {nm:<12}{res.params[j, 0]: .5f}  t={res.tvalues[j, 0]: .2f}")
    fevd = res.fevd(20).decomp[0][-1]
    print(f"\nFEVD of {T} at 20 steps: " + "  ".join(f"{c} {v:.1%}" for c, v in zip(p.cols, fevd)))


# -------------------------------------------------------------------------------------- forecast
def params_for(a, target: str, avail: list[str]) -> Params:
    """--params if given, else <out-dir>/best_params[_CCY].json if it exists, else CLI/defaults."""
    f = Path(a.params) if a.params else Path(a.out_dir) / f"best_params{suffix(target)}.json"
    if a.params is None and not f.exists():
        f = None
    return params_from_args(a, target, avail, f)


def forecast_one(a, df, p: Params, levels: list[float]) -> dict:
    y, x, dates = design(df, p)
    if p.window:
        y, x, dates = y[-p.window:], x[-p.window:], dates[-p.window:]
    res = fit_var(x, p.lags, a.maxlags)
    steps = max(1, round(a.h / BDAYS[p.freq]))
    mean, band = forecast_path(res, x, y[-1], steps, p.transform, a.alpha)
    fan = {lv: forecast_path(res, x, y[-1], steps, p.transform, 1 - lv)[1] for lv in levels}
    import pandas as pd
    k = BDAYS[p.freq]
    fdates = [d.date() for d in pd.bdate_range(pd.Timestamp(dates[-1]) + pd.offsets.BDay(1), periods=steps * k)[k - 1::k]]
    return dict(p=p, k_ar=res.k_ar, y=y, dates=dates, fdates=fdates, mean=mean, band=band, fan=fan, steps=steps)


def plot_fan(ax, r: dict, history: int, alpha: float, i: int = 0, legend: bool = True):
    p, k = r["p"], BDAYS[r["p"].freq]
    c = p.cols[i]
    hist = min(len(r["dates"]), history // k)
    ax.plot(r["dates"][-hist:], r["y"][-hist:, i], color="#1f3b73", lw=1.3, label=f"{c} history")
    fd, m = r["fdates"], r["mean"][:, i]
    for n, (lv, hw) in enumerate(sorted(r["fan"].items(), reverse=True)):
        ax.fill_between(fd, m - hw[:, i], m + hw[:, i], color="#c0392b", alpha=0.12 + 0.08 * n, lw=0,
                        label=f"{lv:.0%} band")
    ax.plot(fd, m, color="#c0392b", lw=1.6, label="VAR forecast")
    ax.axvline(r["dates"][-1], color="0.5", lw=0.8, ls=":")
    lo = m[-1] - r["band"][-1, i]
    hi = m[-1] + r["band"][-1, i]
    ax.annotate(f"{m[-1]:.2f}%\n[{lo:.2f}, {hi:.2f}]", (fd[-1], m[-1]), xytext=(4, 0), textcoords="offset points",
                fontsize=7, va="center", color="#c0392b")
    ax.set_ylabel("%")
    ax.margins(x=0.1)                       # room for the end-of-horizon label
    ax.grid(alpha=0.3)
    if legend:
        ax.legend(loc="upper left", fontsize=7)


def cmd_forecast(a):
    df = load_panel(a.data, a.target.upper())
    avail = ccys_of(df)
    levels = sorted(set(a.bands) | {1 - a.alpha})
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    runs, summary = {}, []
    for t in targets_of(a, df):
        r = forecast_one(a, df, params_for(a, t, avail), levels)
        runs[t] = r
        p, mean, band, steps, y = r["p"], r["mean"], r["band"], r["steps"], r["y"]
        out = {"date": r["fdates"]}
        for i, c in enumerate(p.cols):
            out[c] = mean[:, i]
            out[f"{c}_lo"] = mean[:, i] - band[:, i]
            out[f"{c}_hi"] = mean[:, i] + band[:, i]
        pl.DataFrame(out).write_csv(od / f"forecast{suffix(t)}.csv")
        unit = "d" if p.freq == "D" else "w"
        print(f"{t}  {p.label()}  fitted p={r['k_ar']} on {len(r['dates'])} obs to {r['dates'][-1]}")
        print(f"  last {y[-1, 0]:.3f}%  ->  " + "  ".join(
            f"+{n}{unit}: {mean[n - 1, 0]:.3f} [{mean[n - 1, 0] - band[n - 1, 0]:.3f}, {mean[n - 1, 0] + band[n - 1, 0]:.3f}]"
            for n in sorted({1, max(1, steps // 4), max(1, steps // 2), steps})))
        summary.append({"ccy": t, "params": p.label(), "last": y[-1, 0], "fc_end": mean[-1, 0],
                        "lo": mean[-1, 0] - band[-1, 0], "hi": mean[-1, 0] + band[-1, 0], "end_date": r["fdates"][-1]})
        fig, ax = plt.subplots(figsize=(11, 5))
        plot_fan(ax, r, a.history, a.alpha)
        ax.set_title(f"{t} rate forecast, {steps} {'business days' if p.freq == 'D' else 'weeks'} ahead  ({p.label()})",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(od / f"forecast_{t.lower()}.png", dpi=150)
        plt.close(fig)
        print(f"  -> {od / f'forecast{suffix(t)}.csv'}, {od / f'forecast_{t.lower()}.png'}")

    if len(runs) > 1:
        n = len(runs)
        cols = 2 if n <= 4 else 3
        rows_ = -(-n // cols)
        fig, axs = plt.subplots(rows_, cols, figsize=(6.2 * cols, 3.6 * rows_), squeeze=False)
        for ax, (t, r) in zip(axs.flat, runs.items()):
            plot_fan(ax, r, a.history, a.alpha, legend=(t == next(iter(runs))))
            ax.set_title(f"{t}   {r['p'].label()}", fontsize=8)
            ax.tick_params(labelsize=7)
        for ax in list(axs.flat)[n:]:
            ax.set_visible(False)
        fig.suptitle(f"VAR rate forecasts, {a.h} business days ahead "
                     f"(bands {', '.join(f'{lv:.0%}' for lv in sorted(r['fan']))})", fontsize=11)
        fig.tight_layout()
        fig.savefig(od / "forecast_all.png", dpi=140)
        plt.close(fig)
        sm = pl.DataFrame(summary)
        sm.write_csv(od / "forecast_all.csv")
        with pl.Config(tbl_rows=20, tbl_width_chars=200, float_precision=3, fmt_str_lengths=60):
            print(sm)
        print(f"-> {od / 'forecast_all.png'}, {od / 'forecast_all.csv'}")


# -------------------------------------------------------------------------------------- optimize
def walk_forward(df_dict, p: Params, horizon: int, test_start, step: int, maxlags: int, alpha: float):
    """Refit at every `step`-th origin inside the test period and score the HUF level forecast
    `horizon` business days ahead. Returns metrics dict."""
    df = pl.DataFrame(df_dict)
    y, x, dates = design(df, p)
    k = BDAYS[p.freq]
    hs = max(1, round(horizon / k))
    st = max(1, round(step / k))
    first = next((i for i, d in enumerate(dates) if d >= test_start), None)
    if first is None:
        return None
    errs, rw, cover, hits = [], [], [], []
    for t in range(first, len(dates) - hs, st):
        lo = max(0, t + 1 - p.window) if p.window else 0
        xs = x[lo:t + 1]
        try:
            res = fit_var(xs, p.lags, maxlags)
            mean, band = forecast_path(res, xs, y[t], hs, p.transform, alpha)
        except Exception:  # noqa: BLE001  (singular fit in a tiny window etc.)
            continue
        actual = y[t + hs, 0]
        e = mean[-1, 0] - actual
        errs.append(e)
        rw.append(y[t, 0] - actual)
        cover.append(abs(e) <= band[-1, 0])
        if actual != y[t, 0]:
            hits.append(np.sign(mean[-1, 0] - y[t, 0]) == np.sign(actual - y[t, 0]))
    if len(errs) < 5:
        return None
    errs, rw = np.array(errs), np.array(rw)
    rmse, rmse_rw = float(np.sqrt(np.mean(errs ** 2))), float(np.sqrt(np.mean(rw ** 2)))
    return {**{k_: (v if not isinstance(v, tuple) else "+".join(v)) for k_, v in asdict(p).items()},
            "n_fc": len(errs), "rmse_bp": rmse * 100, "mae_bp": float(np.mean(np.abs(errs))) * 100,
            "rmse_rw_bp": rmse_rw * 100, "rel_rmse": rmse / rmse_rw if rmse_rw else np.nan,
            "hit_rate": float(np.mean(hits)) if hits else np.nan, "coverage": float(np.mean(cover)),
            "bias_bp": float(np.mean(errs)) * 100}


def expand_grid(g: dict, available: list[str], target: str = TARGET) -> list[Params]:
    ins = g.get("inputs", ["ALL"])
    sets = []
    for s in ins:
        if s == "ALL":
            sets.append(tuple(available))
        elif s == "SUBSETS":
            for r in range(1, len(available) + 1):
                sets += list(itertools.combinations(available, r))
        elif s == "SINGLE":
            sets += [(c,) for c in available]
        else:
            sets.append(tuple(c.strip().upper() for c in s.split("+") if c.strip().upper() in available))
    sets = list(dict.fromkeys(sets))
    out = []
    for lags, tr, win, fq, s in itertools.product(g.get("lags", [2]), g.get("transform", ["diff"]),
                                                  g.get("window", [0]), g.get("freq", ["D"]), sets):
        if win and isinstance(lags, int) and win <= lags * (len(s) + 1) * 3:
            continue
        out.append(Params(lags, tr, int(win), fq, tuple(s), target))
    return out


def cmd_optimize(a):
    df = load_panel(a.data, a.target.upper())
    best = [optimize_one(a, df, t) for t in targets_of(a, df)]
    if len(best) > 1:
        with pl.Config(tbl_rows=20, tbl_width_chars=200, float_precision=3, fmt_str_lengths=40):
            print("\nBest parameter set per target")
            print(pl.DataFrame(best).select("target", "lags", "transform", "window", "freq",
                                            pl.col("inputs").list.join("+"), "rmse_bp", "rel_rmse"))


def optimize_one(a, df, target: str) -> dict:
    available = [c for c in ccys_of(df) if c != target]
    grid = json.loads(Path(a.grid).read_text()) if a.grid else {}
    combos = expand_grid(grid, available, target)
    test_start = df["date"][-1] - __import__("datetime").timedelta(days=int(a.test_years * 365.25))
    print(f"\n[{target}] {len(combos)} parameter sets  horizon {a.horizon}bd  test from {test_start}  "
          f"step {a.step}bd  {a.workers} {a.executor} workers")
    # one BLAS thread per worker, the parallelism comes from the pool
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    try:
        from threadpoolctl import threadpool_limits
        limiter = threadpool_limits(1)
    except ImportError:
        limiter = None
    dd = df.to_dict(as_series=False)
    Pool = ThreadPoolExecutor if a.executor == "thread" else ProcessPoolExecutor
    rows, t0 = [], time.time()
    with Pool(max_workers=a.workers) as ex:
        futs = {ex.submit(walk_forward, dd, p, a.horizon, test_start, a.step, a.maxlags, a.alpha): p for p in combos}
        for n, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if r:
                rows.append(r)
            if n % max(1, len(combos) // 20) == 0 or n == len(combos):
                best = min(rows, key=lambda z: z["rmse_bp"]) if rows else None
                print(f"  {n}/{len(combos)}  {time.time() - t0:6.1f}s  best rel_rmse "
                      f"{best['rel_rmse']:.3f}" if best else f"  {n}/{len(combos)}")
    if limiter:
        limiter.unregister() if hasattr(limiter, "unregister") else None
    if not rows:
        sys.exit(f"[{target}] no parameter set produced forecasts (test period too short?)")
    res = pl.DataFrame(rows).sort("rmse_bp")
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    sx = suffix(target)
    res.write_csv(od / f"optimize{sx}_h{a.horizon}.csv")
    b = res.row(0, named=True)
    best = {"target": target, "lags": b["lags"], "transform": b["transform"], "window": b["window"],
            "freq": b["freq"], "inputs": b["inputs"].split("+") if b["inputs"] else [], "horizon": a.horizon,
            "rmse_bp": b["rmse_bp"], "rel_rmse": b["rel_rmse"]}
    (od / f"best_params{sx}.json").write_text(json.dumps(best, indent=2))
    with pl.Config(tbl_rows=15, tbl_cols=20, tbl_width_chars=200, float_precision=3):
        print(res.drop("target").head(a.top))
    print(f"\n[{target}] {time.time() - t0:.1f}s  -> {od / f'optimize{sx}_h{a.horizon}.csv'}, "
          f"{od / f'best_params{sx}.json'}")
    print("rel_rmse < 1 means the VAR beat a random walk (no-change forecast) at this horizon.")
    return best


# ------------------------------------------------------------------------------------------- cli
def params_from_args(a, target: str = TARGET, avail: list[str] | None = None, pfile=None) -> Params:
    """CLI flags override the params file; default inputs = every other currency in the panel."""
    pfile = pfile if pfile is not None else getattr(a, "params", None)
    d = json.loads(Path(pfile).read_text()) if pfile else {}
    if d.get("target", target) != target:      # a params file tuned for another target: ignore its inputs
        d = {k: v for k, v in d.items() if k != "inputs"}
    lags = a.lags if a.lags is not None else d.get("lags", 2)
    lags = int(lags) if str(lags).isdigit() else lags
    others = [c for c in (avail or [TARGET, "EUR", "USD", "JPY", "GBP", "CHF", "AUD"]) if c != target]
    inputs = a.inputs or d.get("inputs") or others
    inputs = [c.upper() for c in inputs if c.upper() != target] or others
    return Params(lags, a.transform or d.get("transform", "diff"),
                  a.window if a.window is not None else int(d.get("window", 0)),
                  a.freq or d.get("freq", "D"), tuple(inputs), target)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("fit", "forecast", "optimize"):
        s = sub.add_parser(name)
        s.add_argument("--data", required=True)
        s.add_argument("--maxlags", type=int, default=15)
        s.add_argument("--alpha", type=float, default=0.05)
        s.add_argument("--out-dir", default="rates_VectorAutoRegression/results")
        s.add_argument("--target", default=TARGET, help="currency to model (panel column) or ALL")
        if name != "optimize":
            s.add_argument("--params", help="best_params.json from optimize")
            s.add_argument("--lags", help="int or aic/bic/hqic/fpe")
            s.add_argument("--transform", choices=["diff", "level"])
            s.add_argument("--window", type=int)
            s.add_argument("--freq", choices=["D", "W"])
            s.add_argument("--inputs", nargs="+")
        if name == "forecast":
            s.add_argument("--h", type=int, default=126, help="horizon in business days (126 = half a year)")
            s.add_argument("--bands", type=float, nargs="+", default=[0.5, 0.8, 0.95],
                           help="confidence levels of the fan-chart bands")
            s.add_argument("--history", type=int, default=500, help="business days of history on the chart")
        if name == "optimize":
            s.add_argument("--grid", help="JSON grid (see rates_VectorAutoRegression_grid.json)")
            s.add_argument("--horizon", type=int, default=20, help="business days ahead to score")
            s.add_argument("--test-years", type=float, default=2.0)
            s.add_argument("--step", type=int, default=5, help="business days between refits")
            s.add_argument("--workers", type=int, default=os.cpu_count())
            s.add_argument("--executor", choices=["thread", "process"], default="process")
            s.add_argument("--top", type=int, default=15, help="rows of the ranking to print")
    a = ap.parse_args(argv)
    {"fit": cmd_fit, "forecast": cmd_forecast, "optimize": cmd_optimize}[a.cmd](a)


if __name__ == "__main__":
    main()
