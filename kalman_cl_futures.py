"""CL (WTI crude) futures traded with a Kalman filter — standalone NautilusTrader script.

Model: local linear trend on the log price of the front CL future, on N-minute bars

    level_t = level_{t-1} + slope_{t-1} + w1        y_t = level_t + v     (y = log price)
    slope_t = slope_{t-1}             + w2

Noise scales adapt to the market: the measurement variance R is an EWMA of squared bar
returns (times r-mult), and the process noises are fixed ratios of it (q-level, q-slope).
Small q-slope = a slow, smooth trend estimate; larger = faster and noisier.

Signals (mode):
  trend   z = slope / (its typical size when there is no trend). The scale comes from
          running the same filter on a simulated random walk at start-up, so z is roughly
          standard normal without a trend whatever q-level / q-slope are: z-entry 2 fires
          about 2.5% of the time per side on pure noise.
          Long when z >= z-entry, short when z <= -z-entry; exit when z falls back to
          |z| <= z-exit or flips sign.
  revert  z = (log price - filtered level) / EWMA std of that gap.
          Fade |z| >= z-entry back toward the level; exit when |z| <= z-exit.
Both: protective stop at stop-sigma x (bar sigma x sqrt(bars per session)) from entry,
max-trades-day entries, flat by flat-time.

The filter runs on every bar, around the clock, so it is warm when the session opens; it
resets when the traded contract rolls. Trading follows the CL day session as in
cl_gex_futures.py: entries 09:30-13:45 ET, flat 14:20 before settlement, no new entries
around the Wednesday 10:30 EIA report, none on exchange holidays.

Data: the catalog used for the GEX scripts. Only CL futures are read (quotes for fills,
trades for the bars, or 1-min bars if the catalog has them); options are not needed.

  python kalman_cl_futures.py --catalog fop_data/catalog --start 2026-01-05 --end 2026-03-31 --out results_kalman
  python optimize.py --module kalman_cl_futures --catalog fop_data/catalog --grid kalman_grid.json ...
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from nautilus_trader.config import StrategyConfig  # noqa: E402
from nautilus_trader.model.data import Bar, BarType  # noqa: E402
from nautilus_trader.model.enums import OrderSide  # noqa: E402
from nautilus_trader.model.events import OrderFilled  # noqa: E402
from nautilus_trader.model.identifiers import Venue  # noqa: E402
from nautilus_trader.model.instruments import FuturesContract  # noqa: E402
from nautilus_trader.trading.strategy import Strategy  # noqa: E402

NY = ZoneInfo("America/New_York")
NS_MIN = 60_000_000_000
NS_DAY = 86_400_000_000_000


# ====================================================================== Kalman filter
class LocalLinearTrend:
    """Kalman filter for y_t = level_t + v, level/slope random walks (2-state, scalar obs)."""

    def __init__(self, q_level, q_slope, r_mult=1.0, var_halflife=60, var_floor=1e-10):
        self.q_level, self.q_slope, self.r_mult = q_level, q_slope, r_mult
        self.alpha = 1.0 - 0.5 ** (1.0 / max(var_halflife, 1))
        self.var_floor = var_floor
        self.reset()

    def reset(self):
        self.x = None                     # [level, slope]
        self.P = None
        self.var = None                   # EWMA variance of 1-bar returns
        self.prev_y = None
        self.n = 0
        self.innov = 0.0
        self.S = float("nan")

    def update(self, y):
        """Feed one observation (log price). Returns True once the filter is initialised."""
        if self.prev_y is not None:
            r = y - self.prev_y
            self.var = r * r if self.var is None else (1 - self.alpha) * self.var + self.alpha * r * r
        self.prev_y = y
        if self.x is None:
            self.x = np.array([y, 0.0])
            self.P = np.diag([1e-4, 1e-6])
            self.n = 1
            return False
        R = max(self.var or self.var_floor, self.var_floor) * self.r_mult
        base = max(self.var or self.var_floor, self.var_floor)
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        Q = np.diag([self.q_level * base, self.q_slope * base])
        x = F @ self.x
        P = F @ self.P @ F.T + Q
        S = P[0, 0] + R
        K = P[:, 0] / S
        self.innov = y - x[0]
        self.S = S
        self.x = x + K * self.innov
        self.P = P - np.outer(K, P[0, :])
        self.P = 0.5 * (self.P + self.P.T)
        self.n += 1
        return True

    @property
    def level(self):
        return float(self.x[0])

    @property
    def slope(self):
        return float(self.x[1])

    @property
    def slope_z(self):
        """Calibrated trend z: slope / (bar sigma x slope std on a driftless random walk)."""
        c = self.null_slope_scale()
        return float(self.x[1] / (max(self.sigma, 1e-12) * c))

    def null_slope_scale(self, n=20000, seed=12345):
        """Std of the slope estimate per unit return sigma when there is no trend.

        The filter is scale-free (Q and R move with the return variance), so this depends
        only on q-level, q-slope, r-mult and var-halflife; it is computed once by running
        the same filter on a simulated random walk.
        """
        if getattr(self, "_null", None) is None:
            sim = LocalLinearTrend(self.q_level, self.q_slope, self.r_mult,
                                   round(math.log(0.5) / math.log(1 - self.alpha)))
            sim._null = 1.0
            y = np.cumsum(np.random.default_rng(seed).standard_normal(n))
            sl = []
            for i, v in enumerate(y):
                if sim.update(v) and i > 500:
                    sl.append(sim.slope / max(sim.sigma, 1e-12))
            self._null = float(np.std(sl))
        return self._null

    @property
    def sigma(self):
        return math.sqrt(max(self.var or 0.0, 0.0))


# ====================================================================== helpers
def nyse_holidays(year):
    """US full-day holidays (observed dates); NYMEX has no day session on these days."""
    from dateutil.easter import easter

    def nth(month, weekday, n):
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
    """CLK6 / CLZ25 style outright for `root` (not spreads, not other products)."""
    return re.fullmatch(rf"{re.escape(root)}[FGHJKMNQUVXZ]\d{{1,2}}", raw_symbol) is not None


def _hm(s):
    h, m = s.split(":")
    return time(int(h), int(m))


# ====================================================================== strategy
class KalmanClFuturesConfig(StrategyConfig, frozen=True):
    venue: str = "GLBX"
    root: str = "CL"
    bar_source: str = "INTERNAL"       # INTERNAL: bars from trades; EXTERNAL: 1-min bars in catalog
    trade_start_date: str = "2026-01-05"
    # filter
    bar_minutes: int = 5               # bar size the filter runs on
    mode: str = "trend"                # trend | revert
    q_level: float = 0.05              # level noise / return variance
    q_slope: float = 0.0005            # slope noise / return variance (smaller = smoother trend)
    r_mult: float = 1.0                # measurement noise / return variance
    var_halflife: int = 60             # bars, EWMA half-life of the return variance
    warmup_bars: int = 60              # bars after a reset before trading
    # signals
    z_entry: float = 2.0
    z_exit: float = 0.5
    stop_sigma: float = 1.0            # stop distance in session sigmas (0 = no stop)
    # session (New York time)
    session_open: str = "09:00"
    session_close: str = "14:30"
    trade_start: str = "09:30"
    last_entry: str = "13:45"
    flat_time: str = "14:20"
    max_trades_day: int = 3
    eia_block: str = "10:20-10:50"     # no new entries on eia_weekday in this window ("" = off)
    eia_weekday: int = 2
    roll_days: int = 8
    # sizing
    contracts: int = 1
    output_dir: str = "results_kalman"


class KalmanClFutures(Strategy):

    def __init__(self, config: KalmanClFuturesConfig):
        super().__init__(config)
        self.c = c = config
        if c.mode not in ("trend", "revert"):
            raise ValueError(f"mode must be trend or revert, not {c.mode!r}")
        self.venue = Venue(c.venue)
        self.first_day = pd.Timestamp(c.trade_start_date).date()
        self.t_open, self.t_close = _hm(c.session_open), _hm(c.session_close)
        self.t_trade, self.t_last, self.t_flat = _hm(c.trade_start), _hm(c.last_entry), _hm(c.flat_time)
        self.eia = tuple(_hm(x) for x in c.eia_block.split("-")) if c.eia_block else None
        sess_min = (self.t_close.hour * 60 + self.t_close.minute) - (self.t_open.hour * 60 + self.t_open.minute)
        self.bars_per_session = max(sess_min / c.bar_minutes, 1.0)
        self.kf = LocalLinearTrend(c.q_level, c.q_slope, c.r_mult, c.var_halflife)
        self.gap_var = None                            # revert mode: EWMA of (y - level)^2
        self.gap_alpha = 1.0 - 0.5 ** (1.0 / max(c.var_halflife, 1))
        self.futs: dict[str, FuturesContract] = {}
        self.cur_fut = None
        self.pos = None
        self.trades, self.daily = [], []
        self.day = None
        self.holidays: dict[int, set] = {}
        self.last_bar_ns = 0
        self.start_equity = None
        self.mult = 1000.0
        self.last_z = float("nan")

    # ------------------------------------------------------------------ setup
    def on_start(self):
        for i in self.cache.instruments(venue=self.venue):
            if type(i) is FuturesContract and is_root_future(i.raw_symbol.value, self.c.root):
                self.futs[i.raw_symbol.value] = i
        if not self.futs:
            self.log.error(f"No {self.c.root} futures in the catalog")
            self.stop()
            return
        self.mult = float(next(iter(self.futs.values())).multiplier)
        for f in self.futs.values():
            self.subscribe_bars(self._bar_type(f.id))
        step = timedelta(minutes=5)
        first = pd.Timestamp(self.clock.utc_now()).floor("5min") + step
        self.clock.set_timer("guard", step, start_time=first.to_pydatetime(), callback=self._on_guard)
        self.log.info(f"{self.c.root}: futures {sorted(self.futs)}, {self.c.bar_minutes}-min bars "
                      f"({self.c.bar_source}), mode {self.c.mode}, multiplier {self.mult:g}")

    def _bar_type(self, iid):
        n = int(self.c.bar_minutes)
        if self.c.bar_source == "EXTERNAL" and n > 1:
            return BarType.from_str(f"{iid}-{n}-MINUTE-LAST-INTERNAL@1-MINUTE-EXTERNAL")
        return BarType.from_str(f"{iid}-{n}-MINUTE-LAST-{self.c.bar_source}")

    # ------------------------------------------------------------------ helpers
    def _ny(self, ns=None):
        ns = self.clock.timestamp_ns() if ns is None else ns
        return pd.Timestamp(ns, tz="UTC").tz_convert(NY).to_pydatetime().replace(tzinfo=None)

    def _trade_future(self):
        now = self.clock.timestamp_ns()
        live = [f for f in self.futs.values() if f.expiration_ns > now + self.c.roll_days * NS_DAY]
        if not live:
            live = [f for f in self.futs.values() if f.expiration_ns > now + NS_DAY]
        return min(live, key=lambda f: f.expiration_ns) if live else None

    def _equity(self):
        acc = self.portfolio.account(self.venue)
        if acc is None:
            return float("nan")
        eq = float(acc.balance_total(acc.base_currency).as_double())
        for m in (self.portfolio.unrealized_pnls(self.venue) or {}).values():
            eq += float(m.as_double())
        return eq

    def _record_day(self, new_date):
        if self.day is not None and self.day["date"] >= self.first_day:
            self.daily.append(dict(date=self.day["date"], equity=self._equity(), z=self.last_z,
                                   bar_sigma=self.kf.sigma, contract=self.day["contract"],
                                   trades=self.day["trades"]))
        self.day = None if new_date is None else dict(date=new_date, trades=0,
                                                      contract=self.cur_fut.raw_symbol.value
                                                      if self.cur_fut else None)

    # ------------------------------------------------------------------ guard timer
    def _on_guard(self, _event):
        """Flat-time and halt exits that must not depend on bars arriving."""
        if not self.pos or self.pos.get("reason"):
            return
        now = self._ny()
        quiet = self.clock.timestamp_ns() - self.last_bar_ns > (self.c.bar_minutes + 15) * NS_MIN
        if now.time() >= self.t_flat or now.time() < self.t_open:
            self._exit("eod")
        elif quiet:
            self._exit("halt")

    # ------------------------------------------------------------------ bars
    def on_bar(self, bar: Bar):
        tf = self._trade_future()
        if tf is None or bar.bar_type.instrument_id != tf.id or float(bar.volume) <= 0:
            return
        if self.cur_fut is None or tf.id != self.cur_fut.id:
            if self.pos and not self.pos.get("reason"):
                self._exit("roll")
            self.cur_fut = tf
            self.kf.reset()
            self.gap_var = None
            self.log.info(f"filter reset on {tf.raw_symbol.value}")
        self.last_bar_ns = bar.ts_event
        px = float(bar.close)
        if px <= 0:
            return
        y = math.log(px)
        ready = self.kf.update(y)
        gap = y - self.kf.level
        if ready:
            self.gap_var = gap * gap if self.gap_var is None else \
                (1 - self.gap_alpha) * self.gap_var + self.gap_alpha * gap * gap
        if not ready or self.kf.n < self.c.warmup_bars:
            return
        z = self.kf.slope_z if self.c.mode == "trend" else \
            gap / math.sqrt(self.gap_var) if self.gap_var and self.gap_var > 0 else float("nan")
        self.last_z = z

        t = self._ny(bar.ts_event)
        if not (self.t_open < t.time() <= self.t_close):
            return                                       # overnight: filter only
        if self.day is None or self.day["date"] != t.date():
            self._record_day(t.date())
        if t.date() < self.first_day:
            return
        if self.start_equity is None:
            self.start_equity = self._equity()

        if t.time() >= self.t_flat:
            if self.pos and not self.pos.get("reason"):
                self._exit("eod")
            return
        if self.pos:
            if not self.pos.get("reason"):
                self._manage(px, bar, z)
            return
        if t.year not in self.holidays:
            self.holidays[t.year] = nyse_holidays(t.year)
        if t.time() < self.t_trade or t.time() > self.t_last or t.date() in self.holidays[t.year] \
                or self.day["trades"] >= self.c.max_trades_day or not np.isfinite(z) \
                or (self.eia is not None and t.weekday() == self.c.eia_weekday
                    and self.eia[0] <= t.time() <= self.eia[1]):
            return
        side = 0
        if z >= self.c.z_entry:
            side = +1 if self.c.mode == "trend" else -1
        elif z <= -self.c.z_entry:
            side = -1 if self.c.mode == "trend" else +1
        if side:
            self._enter(side, px, z)

    # ------------------------------------------------------------------ orders
    def _enter(self, side, px, z):
        n = int(self.c.contracts)
        if n <= 0:
            return
        sess_sigma = self.kf.sigma * math.sqrt(self.bars_per_session) * px
        stop = px - side * self.c.stop_sigma * sess_sigma if self.c.stop_sigma > 0 else None
        self.pos = dict(side=side, iid=self.cur_fut.id, n=n, stop=stop, opened=self._ny(), z=z,
                        cash=0.0, level=math.exp(self.kf.level), sigma=self.kf.sigma)
        self.day["trades"] += 1
        self.submit_order(self.order_factory.market(
            self.cur_fut.id, OrderSide.BUY if side > 0 else OrderSide.SELL,
            self.cur_fut.make_qty(n), tags=[f"open:{self.c.mode}"]))

    def _exit(self, reason):
        p = self.pos
        p["reason"] = reason
        q = int(self.portfolio.net_position(p["iid"]))
        if q == 0:
            return
        inst = self.cache.instrument(p["iid"])
        self.submit_order(self.order_factory.market(
            p["iid"], OrderSide.SELL if q > 0 else OrderSide.BUY, inst.make_qty(abs(q)),
            tags=[f"close:{reason}"]))

    def _manage(self, px, bar, z):
        p = self.pos
        if int(self.portfolio.net_position(p["iid"])) == 0:
            return                                       # entry not filled yet
        side = p["side"]
        if p["stop"] is not None:
            if (side > 0 and float(bar.low) <= p["stop"]) or (side < 0 and float(bar.high) >= p["stop"]):
                return self._exit("stop")
        if not np.isfinite(z):
            return
        if self.c.mode == "trend":
            if side * z <= self.c.z_exit:                # trend faded or flipped
                self._exit("signal")
        elif -side * z <= self.c.z_exit:                 # back near the filtered level
            self._exit("target")

    def on_order_filled(self, e: OrderFilled):
        p = self.pos
        if p is None or e.instrument_id != p["iid"]:
            return
        signed = float(e.last_qty) * (1 if e.order_side == OrderSide.BUY else -1)
        p["cash"] -= signed * float(e.last_px)
        p.setdefault("entry_px", float(e.last_px))
        if p.get("reason") and int(self.portfolio.net_position(p["iid"])) == 0:
            pnl = p["cash"] * float(self.cache.instrument(p["iid"]).multiplier)
            self.trades.append(dict(kind=self.c.mode, side="long" if p["side"] > 0 else "short",
                                    opened=p["opened"], closed=self._ny(), reason=p["reason"],
                                    n=p["n"], entry=p["entry_px"], exit=float(e.last_px),
                                    z_entry=p["z"], level=p["level"], stop=p["stop"],
                                    pnl=pnl, contract=str(p["iid"])))
            self.log.info(f"{'L' if p['side'] > 0 else 'S'} {p['opened']:%Y-%m-%d %H:%M} -> "
                          f"{self._ny():%H:%M} {p['reason']:6s} n={p['n']} pnl=${pnl:,.0f}")
            self.pos = None

    def on_stop(self):
        self._record_day(None)
        t = pd.DataFrame(self.trades)
        eq = self._equity()
        stats = {"Total P&L (incl. fees)": f"{eq - float(self.start_equity or eq):,.0f}",
                 "Trades": len(t),
                 "Win rate": f"{(t['pnl'] > 0).mean():.0%}" if len(t) else "n/a",
                 "Avg trade P&L": f"{t['pnl'].mean():,.0f}" if len(t) else "n/a",
                 "Long / short": f"{(t['side'] == 'long').sum()} / {(t['side'] == 'short').sum()}" if len(t) else "0 / 0"}
        if len(t):
            for r, g in t.groupby("reason"):
                stats[f"Exit {r}"] = f"{len(g)} trades, {g['pnl'].sum():,.0f}"
        for k, v in stats.items():
            self.log.info(f"{k}: {v}")
        os.makedirs(self.c.output_dir, exist_ok=True)
        t.to_csv(os.path.join(self.c.output_dir, "trades.csv"), index=False)
        pd.DataFrame(self.daily).to_csv(os.path.join(self.c.output_dir, "signals.csv"), index=False)
        pd.Series(stats).to_csv(os.path.join(self.c.output_dir, "summary.csv"), header=False)


# ====================================================================== runner
FIELDS = list(KalmanClFuturesConfig.__struct_fields__)
DEFAULTS = dict(zip(FIELDS, KalmanClFuturesConfig.__struct_defaults__))


def parse_params(pairs, json_path):
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


def has_external_bars(catalog_path, futs):
    bar_dir = os.path.join(catalog_path, "data", "bar")
    if not os.path.isdir(bar_dir):
        return False
    names = os.listdir(bar_dir)
    return any(n.startswith(f"{f.id}-1-MINUTE-LAST-EXTERNAL") for f in futs for n in names)


def build_config(catalog_path, start, end, cash, fee, params, out_dir, venue="GLBX",
                 oi_path=None, chunk_size=None, log_level="INFO", bar_source="auto",
                 warmup_days=4, **_ignored):
    from nautilus_trader.config import (BacktestDataConfig, BacktestEngineConfig,
                                        BacktestRunConfig, BacktestVenueConfig, CacheConfig,
                                        DataEngineConfig, ImportableFeeModelConfig,
                                        ImportableStrategyConfig, LoggingConfig)
    from nautilus_trader.model.data import QuoteTick, TradeTick
    from nautilus_trader.persistence.catalog import ParquetDataCatalog

    root = params.get("root", DEFAULTS["root"])
    futs = [i for i in ParquetDataCatalog(catalog_path).instruments()
            if type(i) is FuturesContract and i.id.venue.value == venue
            and is_root_future(i.raw_symbol.value, root)]
    if not futs:
        raise SystemExit(f"No {root} futures for venue {venue} in {catalog_path}")
    if bar_source == "auto":
        bar_source = "EXTERNAL" if has_external_bars(catalog_path, futs) else "INTERNAL"
    data_start = (pd.Timestamp(start) - pd.Timedelta(days=warmup_days)).isoformat()
    data_end = (pd.Timestamp(end) + pd.Timedelta(days=1)).isoformat()
    ids = [str(f.id) for f in futs]
    strategy = ImportableStrategyConfig(
        strategy_path="kalman_cl_futures:KalmanClFutures",
        config_path="kalman_cl_futures:KalmanClFuturesConfig",
        config=dict(params, venue=venue, bar_source=bar_source,
                    trade_start_date=str(pd.Timestamp(start).date()), output_dir=out_dir))
    venue_cfg = BacktestVenueConfig(
        name=venue, oms_type="NETTING", account_type="MARGIN", base_currency="USD",
        starting_balances=[f"{cash:.2f} USD"],
        fee_model=ImportableFeeModelConfig(
            fee_model_path="nautilus_trader.backtest.models:PerContractFeeModel",
            config_path="nautilus_trader.backtest.config:PerContractFeeModelConfig",
            config={"commission": f"{fee:.2f} USD"}))
    data = [BacktestDataConfig(catalog_path=catalog_path, data_cls=QuoteTick, instrument_ids=ids,
                               start_time=data_start, end_time=data_end)]
    if bar_source == "EXTERNAL":
        data.append(BacktestDataConfig(catalog_path=catalog_path, data_cls=Bar,
                                       bar_types=[f"{i}-1-MINUTE-LAST-EXTERNAL" for i in ids],
                                       start_time=data_start, end_time=data_end))
    else:
        data.append(BacktestDataConfig(catalog_path=catalog_path, data_cls=TradeTick,
                                       instrument_ids=ids, start_time=data_start, end_time=data_end))
    cfg = BacktestRunConfig(
        venues=[venue_cfg], data=data,
        engine=BacktestEngineConfig(
            strategies=[strategy], logging=LoggingConfig(log_level=log_level),
            cache=CacheConfig(tick_capacity=10, bar_capacity=10),
            data_engine=DataEngineConfig(time_bars_build_with_no_updates=False)),
        chunk_size=chunk_size, raise_exception=True, dispose_on_completion=False)
    info = dict(futures=sorted(f.raw_symbol.value for f in futs), bar_source=bar_source,
                data_start=data_start)
    return cfg, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--start", default="2026-01-05",
                    help="first trading day; data from --warmup-days earlier warms the filter")
    ap.add_argument("--end", default="2026-03-31")
    ap.add_argument("--warmup-days", type=int, default=4)
    ap.add_argument("--cash", type=float, default=1_000_000.0)
    ap.add_argument("--fee", type=float, default=2.0, help="commission per contract, USD")
    ap.add_argument("--param", action="append", help="key=value (dashes ok), repeatable")
    ap.add_argument("--params-json", default=None)
    ap.add_argument("--venue", default="GLBX")
    ap.add_argument("--bars", default="auto", choices=["auto", "EXTERNAL", "INTERNAL"])
    ap.add_argument("--chunk-size", type=int, default=250_000,
                    help="records loaded per batch; lower = less memory (default 250,000)")
    ap.add_argument("--out", default="results_kalman")
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args()

    from nautilus_trader.backtest.node import BacktestNode

    params, _ = parse_params(a.param, a.params_json)
    os.makedirs(a.out, exist_ok=True)
    cfg, info = build_config(a.catalog, a.start, a.end, a.cash, a.fee, params, a.out, venue=a.venue,
                             chunk_size=a.chunk_size, log_level=a.log_level, bar_source=a.bars,
                             warmup_days=a.warmup_days)
    print(f"futures {info['futures']}, bars {info['bar_source']}, data from {info['data_start'][:10]}")
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
