"""ES futures traded on dealer gamma exposure (GEX) — standalone NautilusTrader script.

Every `rebalance-minutes` in RTH it rebuilds the ES options chain from cached quotes,
fits a Black-76 smile per expiry and, with open interest, computes:

  net GEX      dealer gamma in $ per 1% move (dealers long calls / short puts)
  flip level   futures price where net GEX changes sign (zero-gamma level)
  cushion      signed distance from spot to the flip, as a fraction of spot
  call / put wall   strikes with the largest call / put gamma exposure

Regimes and trades (ES front future, intraday only, flat by `flat-time`):

  long gamma   cushion >= +gex-buffer. Dealers hedge against the move, so price mean-reverts.
               Fade |z| >= z-entry, where z = (F - session VWAP) / expected move so far
               (front ATM IV). Exit back at VWAP (|z| <= z-exit), stop at |z| >= z-stop.
  short gamma  cushion <= -gex-buffer. Dealers hedge with the move, so moves extend.
               Trade breakouts of the opening range. Stop = mom-stop-sigma x daily sigma,
               trailed; optional take-profit at the call wall (long) / put wall (short).
  neutral      otherwise: no new trades.

Open positions are closed when the regime flips (exit-on-regime-change) and at flat-time.

Data: a Nautilus catalog with ES futures + options instruments and quotes, plus either
futures 1-min bars or futures trades (bars are then built from the trades), and an
open_interest.parquet made from Databento `statistics` (--build-oi). Other products in the
same catalog (CL, LO, ...) are ignored. Without open interest there is no GEX, so no trades.

Open interest
  python es_gex_futures.py --catalog ./catalog --build-oi ./fop_data/raw/statistics

Run
  python es_gex_futures.py --catalog ./catalog --start 2026-01-02 --end 2026-03-31 \
      --param z-entry=1.2 --param contracts=2 --out results_gex
Optimize (all cores)
  python optimize.py --module es_gex_futures --catalog ./catalog --grid gex_grid.json ...
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from bisect import bisect_right
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from nautilus_trader.config import StrategyConfig  # noqa: E402
from nautilus_trader.model.data import Bar, BarType  # noqa: E402
from nautilus_trader.model.enums import OptionKind, OrderSide  # noqa: E402
from nautilus_trader.model.events import OrderFilled  # noqa: E402
from nautilus_trader.model.identifiers import Venue  # noqa: E402
from nautilus_trader.model.instruments import FuturesContract, OptionContract  # noqa: E402
from nautilus_trader.trading.strategy import Strategy  # noqa: E402

NY = ZoneInfo("America/New_York")
MULT = 50.0
LONG_G, SHORT_G, NEUTRAL = "long_gamma", "short_gamma", "neutral"

# ====================================================================== pricing & GEX math
_SQRT2 = math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def norm_cdf(x):
    """Standard normal CDF, Abramowitz-Stegun 7.1.26 (|err| < 1.5e-7)."""
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


def black76_gamma(F, K, T, vol, df):
    d1, sT = _d1(F, K, T, vol)
    return df * norm_pdf(d1) / (F * sT)


def implied_vol(price, F, K, T, is_call, df, iters=40):
    """Vectorized Black-76 IV: Newton steps inside a shrinking bisection bracket (NaN if invalid)."""
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
    """Weighted quadratic iv(x), x = ln(K/F)/sqrt(T), dropping > k_mad*MAD outliers."""
    keep = np.isfinite(iv) & np.isfinite(x) & (w > 0)
    coef = None
    for _ in range(rounds):
        if keep.sum() < 5:
            return None
        coef = np.polyfit(x[keep], iv[keep], 2, w=np.sqrt(w[keep]))
        resid = iv - np.polyval(coef, x)
        mad = np.median(np.abs(resid[keep])) + 1e-6
        keep = keep & (np.abs(resid) <= k_mad * 1.4826 * mad)
    return coef


def smile_for_expiry(F, T, df, strikes, is_call, bid, ask, max_rel_spread=0.35, min_strikes=8):
    """Smile IV at every strike plus ATM IV for one expiry, or None if too few quotes."""
    strikes, bid, ask = (np.asarray(a, dtype=float) for a in (strikes, bid, ask))
    is_call = np.asarray(is_call, dtype=bool)
    mid = 0.5 * (bid + ask)
    quoted = (bid > 0) & (ask > bid) & ((ask - bid) <= max_rel_spread * mid)
    use = quoted & np.where(is_call, strikes >= F, strikes <= F)          # OTM only
    if use.sum() < min_strikes or T <= 0:
        return None
    iv = np.full(strikes.shape, np.nan)
    iv[use] = implied_vol(mid[use], F, strikes[use], T, is_call[use], df)
    x = np.log(strikes / F) / math.sqrt(T)
    d1, _ = _d1(F, strikes, T, np.where(np.isfinite(iv), iv, 0.2))
    vega = df * F * norm_pdf(d1) * math.sqrt(T)
    w = np.where(use & np.isfinite(iv), vega / np.maximum(ask - bid, 1e-6), 0.0)
    coef = fit_smile(x, iv, w)
    if coef is None:
        return None
    return dict(smile_iv=np.clip(np.polyval(coef, x), 0.01, 3.0),
                atm_iv=float(np.clip(np.polyval(coef, 0.0), 0.01, 3.0)))


def gex_curve(items, xs, mult=MULT):
    """Net dealer GEX ($ per 1% move) for relative futures moves xs (sticky-strike vols).

    Convention: customers are net long puts and short calls, so dealers are long call
    gamma (+) and short put gamma (-).
    """
    xs = np.asarray(xs, dtype=float)
    total = np.zeros_like(xs)
    for it in items:
        oi = np.asarray(it["oi"], dtype=float)
        use = oi > 0
        if not use.any():
            continue
        K, iv = it["strikes"][use], it["iv"][use]
        w = np.where(it["is_call"][use], 1.0, -1.0) * oi[use]
        Fx = it["F"] * (1.0 + xs)[:, None]
        g = black76_gamma(Fx, K[None, :], it["T"], iv[None, :], it["df"])
        total += (g * w[None, :]).sum(axis=1) * mult * (it["F"] * (1.0 + xs)) ** 2 * 0.01
    return total


def gamma_cushion(xs, gex):
    """Signed distance to the gamma flip as a fraction of spot (> 0: dealers long gamma)."""
    xs, gex = np.asarray(xs, dtype=float), np.asarray(gex, dtype=float)
    if not np.isfinite(gex).all() or not gex.any():
        return float("nan")
    pos_now = gex[int(np.argmin(np.abs(xs)))] > 0
    s = np.sign(gex)
    crossings = np.flatnonzero(s[:-1] * s[1:] < 0)
    if len(crossings) == 0:
        edge = float(np.max(np.abs(xs)))
        return edge if pos_now else -edge
    xc = [xs[j] - gex[j] * (xs[j + 1] - xs[j]) / (gex[j + 1] - gex[j]) for j in crossings]
    return float(-min(xc, key=abs))


def gamma_walls(items, mult=MULT):
    """(call wall, put wall): strikes with the largest call / put dollar gamma at spot."""
    calls, puts = {}, {}
    for it in items:
        oi = np.asarray(it["oi"], dtype=float)
        use = oi > 0
        if not use.any():
            continue
        K, iv, c = it["strikes"][use], it["iv"][use], it["is_call"][use]
        g = black76_gamma(it["F"], K, it["T"], iv, it["df"]) * oi[use] * mult * it["F"] ** 2 * 0.01
        for k, v, is_c in zip(K, g, c):
            book = calls if is_c else puts
            book[k] = book.get(k, 0.0) + float(v)
    cw = max(calls, key=calls.get) if calls else float("nan")
    pw = max(puts, key=puts.get) if puts else float("nan")
    return cw, pw


def nyse_holidays(year):
    """US equity full-day holidays (observed dates). ES has no RTH session on these days."""
    from dateutil.easter import easter

    def nth(month, weekday, n):                       # n-th weekday of month (n=-1: last)
        days = pd.date_range(f"{year}-{month:02d}-01", periods=31, freq="D")
        days = days[(days.month == month) & (days.dayofweek == weekday)]
        return days[n].date()

    def observed(d):
        d = pd.Timestamp(d)
        if d.dayofweek == 5:
            return (d - pd.Timedelta(days=1)).date()
        if d.dayofweek == 6:
            return (d + pd.Timedelta(days=1)).date()
        return d.date()

    out = {observed(f"{year}-01-01"), nth(1, 0, 2), nth(2, 0, 2),
           (pd.Timestamp(easter(year)) - pd.Timedelta(days=2)).date(),
           nth(5, 0, -1), observed(f"{year}-07-04"), nth(9, 0, 0), nth(11, 3, 3),
           observed(f"{year}-12-25")}
    if year >= 2022:
        out.add(observed(f"{year}-06-19"))
    return out


def is_root_future(raw_symbol, root):
    """ESH6 / ESZ25 style outright for `root` (not CLH6, not ESH6-ESM6 spreads)."""
    return re.fullmatch(rf"{re.escape(root)}[FGHJKMNQUVXZ]\d{{1,2}}", raw_symbol) is not None


def build_open_interest(stats_paths, out_path):
    """Databento `statistics` DBN files -> parquet(ts_event=ts_recv ns, symbol, oi)."""
    import databento as db

    rows = []
    for p in stats_paths:
        df = db.DBNStore.from_file(p).to_df()
        if df.empty or "stat_type" not in df:
            continue
        df = df[df["stat_type"] == 9]                 # StatType.OPEN_INTEREST
        if df.empty:
            continue
        rows.append(pd.DataFrame({"ts_event": pd.DatetimeIndex(df.index).as_unit("ns").asi8,
                                  "symbol": df["symbol"].astype(str).values,
                                  "oi": df["quantity"].astype(float).values}))
    if not rows:
        raise SystemExit("No open-interest records (stat_type 9) found in the statistics files")
    oi = pd.concat(rows).drop_duplicates().sort_values("ts_event")
    oi.to_parquet(out_path, index=False)
    return oi


def build_volume_oi(catalog_path, opts, start, end, days, out_path, chunk=200):
    """Open-interest proxy from option trades: per option, the traded volume summed over the
    last `days` CME sessions, stamped at the session close (17:00 ET) so a backtest only sees
    it from the next session on (no look-ahead). Same format as the open-interest parquet."""
    from nautilus_trader.persistence.catalog import ParquetDataCatalog

    cat = ParquetDataCatalog(catalog_path)
    t0 = pd.Timestamp(start, tz="UTC") - pd.Timedelta(days=days * 7 // 5 + 7)
    t1 = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    ids = [str(o.id) for o in opts]
    sym_of = {str(o.id): o.raw_symbol.value for o in opts}
    vol = {}                                              # (symbol, session) -> contracts
    shift = 6 * 3_600_000_000_000                         # session opening 18:00 ET -> next day
    for i in range(0, len(ids), chunk):
        try:
            ticks = cat.trade_ticks(instrument_ids=ids[i:i + chunk], start=t0, end=t1)
        except Exception:                                 # noqa: BLE001 (no trades for this chunk)
            continue
        for t in ticks:
            ny = pd.Timestamp(t.ts_event + shift, tz="UTC").tz_convert("America/New_York")
            key = (sym_of[str(t.instrument_id)], ny.date())
            vol[key] = vol.get(key, 0.0) + float(t.size)
    if not vol:
        raise SystemExit("No option trades in the catalog to build a volume proxy from")
    df = pd.Series(vol).rename_axis(["symbol", "session"]).rename("v").reset_index()
    wide = df.pivot_table(index="session", columns="symbol", values="v", aggfunc="sum").sort_index()
    roll = wide.fillna(0.0).rolling(days, min_periods=1).sum()
    out = roll.stack().rename("oi").reset_index()
    out = out[out["oi"] > 0]
    close = pd.to_datetime(out["session"]).dt.tz_localize("America/New_York") + pd.Timedelta(hours=17)
    out = pd.DataFrame({"ts_event": close.dt.tz_convert("UTC").dt.as_unit("ns").astype("int64").values,
                        "symbol": out["symbol"].values, "oi": out["oi"].values})
    tmp = f"{out_path}.tmp{os.getpid()}"                  # atomic: parallel optimizer workers
    out.sort_values("ts_event").to_parquet(tmp, index=False)
    os.replace(tmp, out_path)
    return out


def classify(cushion, buffer):
    if not np.isfinite(cushion):
        return NEUTRAL
    if cushion >= buffer:
        return LONG_G
    if cushion <= -buffer:
        return SHORT_G
    return NEUTRAL


def size_by_risk(equity, stop_pts, risk_pct, max_contracts, mult=MULT):
    if stop_pts <= 0 or equity <= 0:
        return 0
    return int(max(0, min(max_contracts, math.floor(risk_pct * equity / (stop_pts * mult)))))


# ====================================================================== strategy
class EsGexFuturesConfig(StrategyConfig, frozen=True):
    venue: str = "GLBX"
    root: str = "ES"                   # futures root; other products in the catalog are ignored
    bar_source: str = "EXTERNAL"       # EXTERNAL: 1-min bars in the catalog; INTERNAL: built from trades
    trade_start_date: str = "2026-01-01"
    # GEX
    rebalance_minutes: int = 15
    gex_buffer: float = 0.0025         # |cushion| needed to call a regime (0.25% of spot)
    gex_max_dte: int = 60
    gex_range: float = 0.08
    gex_steps: int = 81
    strike_pct: float = 0.13
    max_rel_spread: float = 0.35
    stale_minutes: int = 30
    rate: float = 0.04
    # session
    or_minutes: int = 30               # opening range from 09:30
    trade_start: str = "10:00"
    last_entry: str = "15:00"
    flat_time: str = "15:50"
    max_trades_day: int = 2
    roll_days: int = 8                 # trade the next contract this many days before expiry
    exit_on_regime_change: int = 1
    # long gamma: mean reversion to VWAP
    mr_enabled: int = 1
    z_entry: float = 1.0
    z_exit: float = 0.0
    z_stop: float = 2.0
    rth_var_share: float = 0.75        # share of the daily variance that happens in RTH (ES ~0.7-0.8)
    # short gamma: opening-range breakout
    mom_enabled: int = 1
    mom_stop_sigma: float = 0.5        # stop distance in daily sigmas (from front ATM IV)
    mom_use_walls: int = 1             # take profit at call wall (long) / put wall (short)
    # sizing
    contracts: int = 1
    risk_pct: float = 0.0              # > 0: size so the stop risks this share of equity
    max_contracts: int = 50
    oi_path: str = ""
    oi_source: str = "file"            # file: open interest; volume: traded-volume proxy (same file
                                       # format, made by the runner); unit: every option weight 1
    output_dir: str = "results_gex"


def _hm(s):
    h, m = s.split(":")
    return time(int(h), int(m))


class EsGexFutures(Strategy):

    def __init__(self, config: EsGexFuturesConfig):
        super().__init__(config)
        self.c = c = config
        self.venue = Venue(c.venue)
        self.first_day = pd.Timestamp(c.trade_start_date).date()
        self.t_trade, self.t_last, self.t_flat = _hm(c.trade_start), _hm(c.last_entry), _hm(c.flat_time)
        self.futs: dict[str, FuturesContract] = {}
        self.chain: list[tuple] = []                 # (OptionContract, und raw symbol, strike, is_call, exp_date)
        self.oi: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.sig = None
        self.day = None
        self.pos = None
        self.trades, self.daily = [], []
        self.regime_count = {LONG_G: 0, SHORT_G: 0, NEUTRAL: 0}
        self.gex_runs = self.gex_ok = 0
        self.start_equity = None
        self.last_bar_ns = 0
        self.holidays: dict[int, set] = {}

    # ------------------------------------------------------------------ setup
    def on_start(self):
        insts = self.cache.instruments(venue=self.venue)
        for i in insts:
            if type(i) is FuturesContract and is_root_future(i.raw_symbol.value, self.c.root):
                self.futs[i.raw_symbol.value] = i
        by_exp = sorted(self.futs.values(), key=lambda f: f.expiration_ns)
        skipped = 0
        for i in insts:
            if type(i) is not OptionContract:
                continue
            und = i.underlying
            if und not in self.futs:
                if not und.startswith(self.c.root) or is_root_future(und, self.c.root):
                    skipped += 1              # another product (e.g. LO on CL), or its future is missing
                    continue
                # ES options quoting the product code only -> next ES future after expiry
                cand = [f for f in by_exp if f.expiration_ns >= i.expiration_ns - 86_400_000_000_000]
                if not cand:
                    skipped += 1
                    continue
                und = cand[0].raw_symbol.value
            exp = pd.Timestamp(i.expiration_ns, tz="UTC").tz_convert(NY).date()
            self.chain.append((i, und, float(i.strike_price), i.option_kind == OptionKind.CALL, exp))
        if not self.futs or not self.chain:
            self.log.error("Need ES futures and options in the catalog")
            self.stop()
            return
        for f in self.futs.values():
            self.subscribe_bars(self._bar_type(f.id))
        if self.c.oi_path and os.path.exists(self.c.oi_path):
            df = pd.read_parquet(self.c.oi_path).sort_values("ts_event")
            for sym, g in df.groupby("symbol"):
                self.oi[sym] = (g["ts_event"].to_numpy(np.int64), g["oi"].to_numpy(float))
        if self.c.oi_source == "unit":
            self.log.warning("oi_source=unit: every option weighted 1 -> GEX shows the shape of the "
                             "chain's gamma only, not dealer positioning")
        elif not self.oi:
            self.log.warning("No open interest: GEX cannot be computed and the strategy stays flat")
        step = timedelta(minutes=int(self.c.rebalance_minutes))
        first = pd.Timestamp(self.clock.utc_now()).floor(f"{int(self.c.rebalance_minutes)}min") + step
        self.clock.set_timer("gex", step, start_time=first.to_pydatetime(), callback=self._on_gex_timer)
        exps = sorted({row[4] for row in self.chain})
        self.log.info(f"{self.c.root}: futures {sorted(self.futs)}, {len(self.chain)} options on "
                      f"{len(exps)} expiries ({exps[0]} .. {exps[-1]}), {skipped} other options ignored, "
                      f"OI for {len(self.oi)} symbols, bars {self.c.bar_source}")

    # ------------------------------------------------------------------ helpers
    def _ny(self, ns=None):
        ns = self.clock.timestamp_ns() if ns is None else ns
        return pd.Timestamp(ns, tz="UTC").tz_convert(NY).to_pydatetime().replace(tzinfo=None)

    def _bar_type(self, iid):
        return BarType.from_str(f"{iid}-1-MINUTE-LAST-{self.c.bar_source}")

    def _trade_future(self):
        """Front contract, rolled `roll_days` before expiry (where ES volume moves).

        If the next contract is not in the catalog, keep trading the front one until the
        day before it expires.
        """
        now = self.clock.timestamp_ns()
        day = 86_400_000_000_000
        live = [f for f in self.futs.values() if f.expiration_ns > now + self.c.roll_days * day]
        if not live:
            live = [f for f in self.futs.values() if f.expiration_ns > now + day]
        return min(live, key=lambda f: f.expiration_ns) if live else None

    def _price(self, iid):
        q = self.cache.quote_tick(iid)
        if q is not None and q.bid_price > 0 and q.ask_price > q.bid_price:
            return 0.5 * (float(q.bid_price) + float(q.ask_price))
        b = self.cache.bar(self._bar_type(iid))
        return float(b.close) if b is not None else float("nan")

    def _oi(self, sym, now_ns):
        if self.c.oi_source == "unit":
            return 1.0
        rec = self.oi.get(sym)
        if rec is None:
            return 0.0
        i = bisect_right(rec[0], now_ns) - 1
        return float(rec[1][i]) if i >= 0 else 0.0

    def _equity(self):
        acc = self.portfolio.account(self.venue)
        if acc is None:
            return float("nan")
        eq = float(acc.balance_total(acc.base_currency).as_double())
        for m in (self.portfolio.unrealized_pnls(self.venue) or {}).values():
            eq += float(m.as_double())
        return eq

    # ------------------------------------------------------------------ GEX signal
    def _on_gex_timer(self, _event):
        now = self._ny()
        if self.pos and not self.pos.get("reason"):
            # Exits that must not depend on bars arriving (early closes, halts).
            quiet = self.clock.timestamp_ns() - self.last_bar_ns > 15 * 60_000_000_000
            if now.time() >= self.t_flat or now.time() < time(9, 30):
                self._exit("eod")
            elif quiet:
                self._exit("halt")
        if not (time(9, 30) <= now.time() <= self.t_flat):
            return
        if self.clock.timestamp_ns() - self.last_bar_ns > 5 * 60_000_000_000:
            return                                       # no market today (holiday / weekend)
        sig = self.compute_gex(now)
        if sig is not None:
            self.sig = sig
            if now.date() >= self.first_day:
                self.regime_count[sig["regime"]] += 1

    def compute_gex(self, now):
        now_ns = self.clock.timestamp_ns()
        today = now.date()
        groups = {}
        prices = {}
        for inst, und, k, is_call, exp in self.chain:
            if inst.expiration_ns <= now_ns or not (0 <= (exp - today).days <= self.c.gex_max_dte):
                continue
            if und not in prices:
                f = self.futs.get(und)
                prices[und] = self._price(f.id) if f is not None and f.expiration_ns > now_ns else float("nan")
            F = prices[und]
            if not np.isfinite(F) or abs(k / F - 1) > self.c.strike_pct:
                continue
            groups.setdefault((und, exp), []).append((inst, k, is_call))
        items, front_iv = [], (10_000, float("nan"))
        stale = self.c.stale_minutes * 60_000_000_000
        for (und, exp), rows in groups.items():
            F = prices[und]
            bid, ask = [], []
            for inst, _, _ in rows:
                q = self.cache.quote_tick(inst.id)
                ok = q is not None and now_ns - q.ts_event <= stale
                bid.append(float(q.bid_price) if ok else 0.0)
                ask.append(float(q.ask_price) if ok else 0.0)
            end = datetime(exp.year, exp.month, exp.day, 16, 0)
            T = max((end - now).total_seconds(), 60.0) / (365.0 * 86400.0)
            df = math.exp(-self.c.rate * T)
            strikes = np.array([r[1] for r in rows])
            is_call = np.array([r[2] for r in rows])
            sm = smile_for_expiry(F, T, df, strikes, is_call, bid, ask, self.c.max_rel_spread)
            if sm is None:
                continue
            dte = (exp - today).days
            if dte >= 1 and dte < front_iv[0]:
                front_iv = (dte, sm["atm_iv"])
            oi = np.array([self._oi(r[0].raw_symbol.value, now_ns) for r in rows])
            items.append(dict(F=F, T=T, df=df, strikes=strikes, is_call=is_call,
                              iv=sm["smile_iv"], oi=oi))
        if not items:
            return None
        self.gex_runs += 1
        if not any((it["oi"] > 0).any() for it in items):
            return dict(regime=NEUTRAL, cushion=float("nan"), flip=float("nan"),
                        call_wall=float("nan"), put_wall=float("nan"), gex=float("nan"),
                        atm_iv=front_iv[1], ts=now_ns)
        self.gex_ok += 1
        xs = np.linspace(-self.c.gex_range, self.c.gex_range, int(self.c.gex_steps))
        curve = gex_curve(items, xs)
        cushion = gamma_cushion(xs, curve)
        tf = self._trade_future()
        spot = self._price(tf.id) if tf is not None else float("nan")
        cw, pw = gamma_walls(items)
        return dict(regime=classify(cushion, self.c.gex_buffer), cushion=cushion,
                    flip=spot * (1.0 - cushion) if np.isfinite(cushion) else float("nan"),
                    call_wall=cw, put_wall=pw, gex=float(curve[int(np.argmin(np.abs(xs)))]),
                    atm_iv=front_iv[1], ts=now_ns)

    # ------------------------------------------------------------------ intraday state
    def _new_day(self, d):
        if self.day is not None and self.day["date"] >= self.first_day:
            s = self.sig or {}
            self.daily.append(dict(date=self.day["date"], equity=self._equity(),
                                   regime=s.get("regime"), cushion=s.get("cushion"),
                                   flip=s.get("flip"), call_wall=s.get("call_wall"),
                                   put_wall=s.get("put_wall"), gex_bn_per_1pct=(s.get("gex") or np.nan) / 1e9,
                                   atm_iv=s.get("atm_iv"), trades=self.day["trades"]))
        self.day = dict(date=d, pv=0.0, v=0.0, or_hi=-np.inf, or_lo=np.inf, trades=0)

    def on_bar(self, bar: Bar):
        tf = self._trade_future()
        if tf is None or bar.bar_type.instrument_id != tf.id:
            return
        if float(bar.volume) <= 0:
            return                                       # no trades in this minute (closed market)
        self.last_bar_ns = bar.ts_event
        t = self._ny(bar.ts_event)                     # bar close time
        if not (time(9, 31) <= t.time() <= time(16, 0)):
            return
        if self.day is None or self.day["date"] != t.date():
            self._new_day(t.date())
        day = self.day
        px, vol = float(bar.close), float(bar.volume)
        typical = (float(bar.high) + float(bar.low) + px) / 3.0
        day["pv"] += typical * max(vol, 1.0)
        day["v"] += max(vol, 1.0)
        mins = (t.hour - 9) * 60 + t.minute - 30
        if mins <= self.c.or_minutes:
            day["or_hi"] = max(day["or_hi"], float(bar.high))
            day["or_lo"] = min(day["or_lo"], float(bar.low))
        if t.date() < self.first_day:
            return
        if self.start_equity is None:
            self.start_equity = self._equity()

        if t.time() >= self.t_flat:
            if self.pos and not self.pos.get("reason"):
                self._exit("eod")
            return
        sig = self.sig
        fresh = sig is not None and self.clock.timestamp_ns() - sig["ts"] <= \
            2 * self.c.rebalance_minutes * 60_000_000_000
        vwap = day["pv"] / day["v"]
        iv = sig["atm_iv"] if fresh and np.isfinite(sig.get("atm_iv", np.nan)) else float("nan")
        sd_day = px * iv / math.sqrt(252.0) if np.isfinite(iv) else float("nan")
        unit = sd_day * math.sqrt(self.c.rth_var_share * max(mins, 15) / 390.0)
        z = (px - vwap) / unit if np.isfinite(unit) and unit > 0 else float("nan")

        if self.pos:
            if not self.pos.get("reason"):
                self._manage(px, bar, z, sig if fresh else None, sd_day)
            return
        if t.year not in self.holidays:
            self.holidays[t.year] = nyse_holidays(t.year)
        if not fresh or t.time() < self.t_trade or t.time() > self.t_last \
                or t.date() in self.holidays[t.year] \
                or day["trades"] >= self.c.max_trades_day or not np.isfinite(z):
            return
        if sig["regime"] == LONG_G and self.c.mr_enabled:
            if z >= self.c.z_entry:
                self._enter("mr", -1, tf, px, stop=None, target=vwap, z=z, sd=unit * self.c.z_stop)
            elif z <= -self.c.z_entry:
                self._enter("mr", +1, tf, px, stop=None, target=vwap, z=z, sd=unit * self.c.z_stop)
        elif sig["regime"] == SHORT_G and self.c.mom_enabled and mins > self.c.or_minutes:
            dist = self.c.mom_stop_sigma * sd_day
            if px > day["or_hi"]:
                tgt = sig["call_wall"] if self.c.mom_use_walls and sig["call_wall"] > px else None
                self._enter("mom", +1, tf, px, stop=px - dist, target=tgt, z=z, sd=dist)
            elif px < day["or_lo"]:
                tgt = sig["put_wall"] if self.c.mom_use_walls and sig["put_wall"] < px else None
                self._enter("mom", -1, tf, px, stop=px + dist, target=tgt, z=z, sd=dist)

    # ------------------------------------------------------------------ orders
    def _enter(self, kind, side, fut, px, stop, target, z, sd):
        if self.c.risk_pct > 0:
            n = size_by_risk(self._equity(), sd, self.c.risk_pct, self.c.max_contracts)
        else:
            n = int(self.c.contracts)
        if n <= 0:
            return
        s = self.sig
        self.pos = dict(kind=kind, side=side, iid=fut.id, n=n, stop=stop, target=target,
                        trail=sd, opened=self._ny(), entry_ref=px, cash=0.0, z=z,
                        regime=s["regime"], cushion=s["cushion"], flip=s["flip"],
                        call_wall=s["call_wall"], put_wall=s["put_wall"])
        self.day["trades"] += 1
        o = self.order_factory.market(fut.id, OrderSide.BUY if side > 0 else OrderSide.SELL,
                                      fut.make_qty(n), tags=[f"open:{kind}"])
        self.submit_order(o)

    def _exit(self, reason):
        p = self.pos
        p["reason"] = reason
        q = int(self.portfolio.net_position(p["iid"]))
        if q == 0:
            return
        inst = self.cache.instrument(p["iid"])
        o = self.order_factory.market(p["iid"], OrderSide.SELL if q > 0 else OrderSide.BUY,
                                      inst.make_qty(abs(q)), tags=[f"close:{reason}"])
        self.submit_order(o)

    def _manage(self, px, bar, z, sig, sd_day):
        p = self.pos
        if int(self.portfolio.net_position(p["iid"])) == 0:
            return                                       # entry not filled yet
        side = p["side"]
        if p["kind"] == "mr":
            if np.isfinite(z):
                if side * z >= -self.c.z_exit:           # back to (or through) VWAP
                    return self._exit("target")
                if -side * z >= self.c.z_stop:           # moved further against us
                    return self._exit("stop")
        else:
            hi, lo = float(bar.high), float(bar.low)
            if (side > 0 and lo <= p["stop"]) or (side < 0 and hi >= p["stop"]):
                return self._exit("stop")
            if p["target"] is not None and ((side > 0 and hi >= p["target"])
                                            or (side < 0 and lo <= p["target"])):
                return self._exit("target")
            trail = p["trail"]
            p["stop"] = max(p["stop"], px - trail) if side > 0 else min(p["stop"], px + trail)
        if self.c.exit_on_regime_change and sig is not None and sig["regime"] != p["regime"]:
            self._exit("regime")

    def on_order_filled(self, e: OrderFilled):
        p = self.pos
        if p is None or e.instrument_id != p["iid"]:
            return
        signed = float(e.last_qty) * (1 if e.order_side == OrderSide.BUY else -1)
        p["cash"] -= signed * float(e.last_px)
        if "entry_px" not in p:
            p["entry_px"] = float(e.last_px)
        if p.get("reason") and int(self.portfolio.net_position(p["iid"])) == 0:
            mult = float(self.cache.instrument(p["iid"]).multiplier)
            pnl = p["cash"] * mult
            self.trades.append(dict(kind=p["kind"], side="long" if p["side"] > 0 else "short",
                                    opened=p["opened"], closed=self._ny(), reason=p["reason"],
                                    n=p["n"], entry=p["entry_px"], exit=float(e.last_px),
                                    z_entry=p["z"], regime=p["regime"], cushion=p["cushion"],
                                    flip=p["flip"], call_wall=p["call_wall"], put_wall=p["put_wall"],
                                    pnl=pnl, contract=str(p["iid"])))
            self.log.info(f"{p['kind']:3s} {'L' if p['side'] > 0 else 'S'} {p['opened']:%Y-%m-%d %H:%M} -> "
                          f"{self._ny():%H:%M} {p['reason']:7s} n={p['n']} pnl=${pnl:,.0f}")
            self.pos = None

    def on_stop(self):
        if self.day is not None:
            self._new_day(None)
        t = pd.DataFrame(self.trades)
        eq = self._equity()
        stats = {"Total P&L (incl. fees)": f"{eq - float(self.start_equity or eq):,.0f}",
                 "Trades": len(t)}
        for kind, label in (("mr", "Mean-reversion"), ("mom", "Breakout")):
            k = t[t["kind"] == kind] if len(t) else t
            stats[f"{label} trades"] = len(k)
            stats[f"{label} win rate"] = f"{(k['pnl'] > 0).mean():.0%}" if len(k) else "n/a"
            stats[f"{label} P&L"] = f"{k['pnl'].sum():,.0f}" if len(k) else "0"
        tot = sum(self.regime_count.values()) or 1
        for r in (LONG_G, SHORT_G, NEUTRAL):
            stats[f"Time {r}"] = f"{self.regime_count[r] / tot:.0%}"
        stats["GEX coverage"] = f"{self.gex_ok}/{self.gex_runs}"
        for k, v in stats.items():
            self.log.info(f"{k}: {v}")
        os.makedirs(self.c.output_dir, exist_ok=True)
        t.to_csv(os.path.join(self.c.output_dir, "trades.csv"), index=False)
        pd.DataFrame(self.daily).to_csv(os.path.join(self.c.output_dir, "signals.csv"), index=False)
        pd.Series(stats).to_csv(os.path.join(self.c.output_dir, "summary.csv"), header=False)


# ====================================================================== runner
FIELDS = list(EsGexFuturesConfig.__struct_fields__)
DEFAULTS = dict(zip(FIELDS, EsGexFuturesConfig.__struct_defaults__))


def parse_params(pairs, json_path):
    """Lean-style names (dashes) -> config fields. Returns (params, raw)."""
    raw = {}
    if json_path:
        with open(json_path) as fh:
            raw.update(json.load(fh))
    for kv in pairs or []:
        k, v = kv.split("=", 1)
        raw[k] = v
    out = {}
    for k, v in raw.items():
        name = k.replace("-", "_")
        if name in ("start", "end", "cash"):
            continue
        if name not in DEFAULTS:
            raise SystemExit(f"Unknown parameter: {k}")
        d = DEFAULTS[name]
        out[name] = int(float(v)) if isinstance(d, int) else type(d)(v)
    return out, raw


def select_instruments(catalog, root, venue):
    """ES outright futures plus the options written on them (CL/LO etc. are left out)."""
    insts = [i for i in catalog.instruments() if i.id.venue.value == venue]
    futs = [i for i in insts if type(i) is FuturesContract and is_root_future(i.raw_symbol.value, root)]
    fut_syms = {f.raw_symbol.value for f in futs}
    opts = [i for i in insts if type(i) is OptionContract
            and (i.underlying in fut_syms
                 or (i.underlying.startswith(root) and not is_root_future(i.underlying, root)))]
    return futs, opts


def has_external_bars(catalog_path, futs):
    bar_dir = os.path.join(catalog_path, "data", "bar")
    if not os.path.isdir(bar_dir):
        return False
    names = os.listdir(bar_dir)
    return any(n.startswith(f"{f.id}-1-MINUTE-LAST-EXTERNAL") for f in futs for n in names)


def build_config(catalog_path, start, end, cash, fee, params, out_dir, venue="GLBX",
                 oi_path=None, chunk_size=None, log_level="INFO", bar_source="auto",
                 oi_source=None, oi_volume_days=5):
    from nautilus_trader.config import (BacktestDataConfig, BacktestEngineConfig,
                                        BacktestRunConfig, BacktestVenueConfig,
                                        ImportableFeeModelConfig, ImportableStrategyConfig,
                                        CacheConfig, DataEngineConfig, LoggingConfig)
    from nautilus_trader.model.data import QuoteTick, TradeTick
    from nautilus_trader.persistence.catalog import ParquetDataCatalog

    root = params.get("root", DEFAULTS["root"])
    data_start = (pd.Timestamp(start) - pd.Timedelta(days=1)).isoformat()
    data_end = (pd.Timestamp(end) + pd.Timedelta(days=1)).isoformat()
    futs, opts = select_instruments(ParquetDataCatalog(catalog_path), root, venue)
    if not futs or not opts:
        raise SystemExit(f"No {root} futures/options for venue {venue} in {catalog_path}")
    if bar_source == "auto":
        bar_source = "EXTERNAL" if has_external_bars(catalog_path, futs) else "INTERNAL"
    oi_source = oi_source or params.get("oi_source", DEFAULTS["oi_source"])
    params = dict(params, oi_source="unit" if oi_source == "unit" else "file")
    if oi_source == "volume":
        oi_path = os.path.join(catalog_path, f"volume_oi_{oi_volume_days}d_{start}_{end}.parquet")
        if not os.path.exists(oi_path):
            print(f"building volume proxy for open interest ({oi_volume_days} sessions) ...")
            build_volume_oi(catalog_path, opts, start, end, oi_volume_days, oi_path)
    elif oi_source == "unit":
        oi_path = ""
    elif oi_path is None:
        cand = os.path.join(catalog_path, "open_interest.parquet")
        oi_path = cand if os.path.exists(cand) else ""
    strategy = ImportableStrategyConfig(
        strategy_path="es_gex_futures:EsGexFutures",
        config_path="es_gex_futures:EsGexFuturesConfig",
        config=dict(params, venue=venue, bar_source=bar_source,
                    trade_start_date=str(pd.Timestamp(start).date()),
                    oi_path=oi_path, output_dir=out_dir))
    venue_cfg = BacktestVenueConfig(
        name=venue, oms_type="NETTING", account_type="MARGIN", base_currency="USD",
        starting_balances=[f"{cash:.2f} USD"],
        fee_model=ImportableFeeModelConfig(
            fee_model_path="nautilus_trader.backtest.models:PerContractFeeModel",
            config_path="nautilus_trader.backtest.config:PerContractFeeModelConfig",
            config={"commission": f"{fee:.2f} USD"}))
    fut_ids = [str(f.id) for f in futs]
    data = [BacktestDataConfig(catalog_path=catalog_path, data_cls=QuoteTick,
                               instrument_ids=fut_ids + [str(o.id) for o in opts],
                               start_time=data_start, end_time=data_end)]
    if bar_source == "EXTERNAL":
        data.append(BacktestDataConfig(catalog_path=catalog_path, data_cls=Bar,
                                       bar_types=[f"{f}-1-MINUTE-LAST-EXTERNAL" for f in fut_ids],
                                       start_time=data_start, end_time=data_end))
    else:   # 1-minute bars aggregated by the engine from the futures trades
        data.append(BacktestDataConfig(catalog_path=catalog_path, data_cls=TradeTick,
                                       instrument_ids=fut_ids,
                                       start_time=data_start, end_time=data_end))
    cfg = BacktestRunConfig(venues=[venue_cfg], data=data,
                            engine=BacktestEngineConfig(
                                strategies=[strategy], logging=LoggingConfig(log_level=log_level),
                                # the strategy only reads the latest quote/bar per instrument
                                cache=CacheConfig(tick_capacity=10, bar_capacity=10),
                                data_engine=DataEngineConfig(time_bars_build_with_no_updates=False)),
                            chunk_size=chunk_size, raise_exception=True, dispose_on_completion=False)
    info = dict(futures=sorted(f.raw_symbol.value for f in futs), options=len(opts),
                expiries=sorted({pd.Timestamp(o.expiration_ns, tz="UTC").date() for o in opts}),
                bar_source=bar_source, oi_path=oi_path, oi_source=oi_source)
    return cfg, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--start", default="2026-01-02")
    ap.add_argument("--end", default="2026-03-31")
    ap.add_argument("--cash", type=float, default=1_000_000.0)
    ap.add_argument("--fee", type=float, default=2.0, help="commission per contract, USD")
    ap.add_argument("--param", action="append", help="key=value (dashes ok), repeatable")
    ap.add_argument("--params-json", default=None)
    ap.add_argument("--venue", default="GLBX")
    ap.add_argument("--oi", default=None, help="open-interest parquet (default <catalog>/open_interest.parquet)")
    ap.add_argument("--build-oi", nargs="+", metavar="STATS",
                    help="Databento `statistics` .dbn.zst files or folders: write the open-interest "
                         "parquet (to --oi or <catalog>/open_interest.parquet) and exit")
    ap.add_argument("--oi-source", default="file", choices=["file", "volume", "unit"],
                    help="file: open_interest.parquet; volume: proxy from option trades in the "
                         "catalog (no OI download needed); unit: every option weight 1")
    ap.add_argument("--oi-volume-days", type=int, default=5,
                    help="sessions of traded volume summed for --oi-source volume")
    ap.add_argument("--bars", default="auto", choices=["auto", "EXTERNAL", "INTERNAL"],
                    help="auto: catalog 1-min bars if present, else build them from futures trades")
    ap.add_argument("--chunk-size", type=int, default=250_000,
                    help="records loaded per batch; lower = less memory (default 250,000)")
    ap.add_argument("--out", default="results_gex")
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args()

    if a.build_oi:
        files = []
        for p in a.build_oi:
            if os.path.isdir(p):
                files += sorted(glob.glob(os.path.join(p, "**", "*.dbn*"), recursive=True))
            elif os.path.isfile(p):
                files.append(p)
            else:
                raise SystemExit(
                    f"{p} does not exist. Download the Databento `statistics` schema first, e.g.\n"
                    "  python future_options_downloader_for_nautilus.py download --parents ES.OPT "
                    "--start 2026-01-01 --end 2026-03-31 --schemas statistics --no-underlying")
        if not files:
            raise SystemExit(f"No .dbn / .dbn.zst files in {' '.join(a.build_oi)}")
        out = a.oi or os.path.join(a.catalog, "open_interest.parquet")
        oi = build_open_interest(files, out)
        print(f"open interest: {len(oi):,} rows, {oi['symbol'].nunique()} symbols, "
              f"{pd.Timestamp(oi['ts_event'].min(), tz='UTC'):%Y-%m-%d} .. "
              f"{pd.Timestamp(oi['ts_event'].max(), tz='UTC'):%Y-%m-%d} -> {out}")
        return

    from nautilus_trader.backtest.node import BacktestNode

    params, _ = parse_params(a.param, a.params_json)
    os.makedirs(a.out, exist_ok=True)
    cfg, info = build_config(a.catalog, a.start, a.end, a.cash, a.fee, params, a.out, venue=a.venue,
                             oi_path=a.oi, chunk_size=a.chunk_size, log_level=a.log_level,
                             bar_source=a.bars, oi_source=a.oi_source,
                             oi_volume_days=a.oi_volume_days)
    print(f"futures {info['futures']}, {info['options']} options on {len(info['expiries'])} expiries "
          f"({info['expiries'][0]} .. {info['expiries'][-1]}), bars {info['bar_source']}")
    print(f"GEX weights: {info['oi_source']}" + (f" ({info['oi_path']})" if info["oi_path"] else ""))
    if info["oi_source"] == "file" and not info["oi_path"]:
        print("WARNING: no open_interest.parquet -> no GEX -> the strategy will not trade.\n"
              "         Run --build-oi on Databento `statistics`, or use --oi-source volume.")
    node = BacktestNode(configs=[cfg])
    node.run()
    eng = node.get_engine(cfg.id)
    eng.trader.generate_order_fills_report().to_csv(os.path.join(a.out, "fills.csv"))
    eng.trader.generate_account_report(eng.list_venues()[0]).to_csv(os.path.join(a.out, "account.csv"))
    print(pd.read_csv(os.path.join(a.out, "summary.csv"), header=None).to_string(index=False, header=False))
    print(f"\nResults written to {os.path.abspath(a.out)}")
    node.dispose()


if __name__ == "__main__":
    main()
