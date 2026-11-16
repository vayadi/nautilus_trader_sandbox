"""Market-state (regime switching) futures strategy — standalone NautilusTrader script.

Every bar is classified into a market state from two independent axes, and the state picks
the tactic. Default instrument is CL (WTI crude); root=ES with ES session params also works.

Trend axis (Wilder ADX / DMI + slow EMA slope):
  up / down   ADX >= adx-trend and |slope| >= slope-min, slope = change of the slow EMA over
              slope-bars, in ATRs per bar. With di-confirm 1 the +DI/-DI ordering must agree
              with the slope sign, otherwise the bar counts as neutral.
  range       ADX <= adx-range
  neutral     everything in between (hysteresis band: no new entries)
Volatility axis (percentile of the current value within the last vol-lookback bars):
  vol-measure atr = ATR / price (NATR), rv = realised vol of rv-len log returns.
  high >= vol-high, low <= vol-low, else normal.
A new raw state only becomes the active state after confirm-bars consecutive bars, so a
single noisy bar does not flip tactics. The state is printed as e.g. "up/normal".

Tactics:
  trend    (state up / down) with the trend only.
           trend-entry breakout: close beyond the Donchian high/low of the previous don-len bars.
           trend-entry pullback: bar trades back to the fast EMA but closes on the trend side of
                                 it, with fast EMA on the right side of the slow EMA.
           trend-entry both: either.
           Exit: initial stop stop-atr x ATR, chandelier trail trail-atr x ATR from the best
           close since entry (0 = off), ratcheted at the venue.
  revert   (state range) fade Bollinger extremes: z = (close - SMA) / SD over bb-len bars.
           Long when z <= -revert-z, short when z >= revert-z. Target: z back to -revert-exit
           (long) / +revert-exit (short); 0 = the mean. Stop stop-atr x ATR.
Volatility gates: high-vol flat | trend | revert | all and low-vol flat | trend | revert | all
say which tactics may open in that volatility state. tactics both | trend | revert switches a
whole tactic off (useful for attribution runs).
exit-on-state 1: a trend trade is closed when the state turns to range or the opposite trend,
a revert trade when a trend against it starts, and either when its volatility gate closes.
max-hold-bars is a time stop (0 = off).

Indicators are kept per contract on every bar around the clock, so after a roll the new
contract is already warm (CL deferred months trade for weeks before they become front).
Entries only in the session window (default 09:30-13:45 ET, none around the Wednesday EIA
report or on exchange holidays). Positions are flat by flat-time unless overnight 1, which
holds them overnight but still flattens on Fridays. Positions are closed on a roll.

Sizing: fixed `contracts`, or risk-usd > 0 = floor(risk-usd / (stop distance x multiplier)),
capped at max-contracts.

Outputs in --out: trades.csv, signals.csv (daily equity + state), states.csv (every bar of the
traded contract with its indicators and state, for plotting), summary.csv.

  python market_state_futures.py --catalog fop_data/catalog --start 2026-01-12 --end 2026-03-31 --out results_market_state
  python market_state_futures.py --catalog fop_data/catalog --param trend-entry=pullback --param high-vol=flat ...
  python optimize.py --module market_state_futures --catalog fop_data/catalog --grid market_state_grid.json ...
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import deque
from datetime import timedelta

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from nautilus_trader.config import StrategyConfig  # noqa: E402
from nautilus_trader.model.data import Bar, BarType  # noqa: E402
from nautilus_trader.model.enums import OrderSide  # noqa: E402
from nautilus_trader.model.events import OrderFilled, OrderRejected  # noqa: E402
from nautilus_trader.model.identifiers import Venue  # noqa: E402
from nautilus_trader.model.instruments import FuturesContract  # noqa: E402
from nautilus_trader.trading.strategy import Strategy  # noqa: E402

# Shared session helpers live in the Kalman script; reuse them so the scripts stay in sync.
from kalman_cl_futures import (NS_DAY, NS_MIN, NY, _hm, has_external_bars,  # noqa: E402
                               is_root_future, nyse_holidays)

GATES = ("flat", "trend", "revert", "all")


# ====================================================================== market state
class MarketState:
    """Indicators + state classifier for one contract. Feed every bar with update()."""

    def __init__(self, c):
        self.c = c
        self.a_atr = 1.0 / c.atr_len
        self.a_adx = 1.0 / c.adx_len
        self.a_f = 2.0 / (c.ema_fast + 1)
        self.a_s = 2.0 / (c.ema_slow + 1)
        self.bars_per_year = 252 * 23 * 60 / c.bar_minutes
        self.n = 0
        self.prev = None                       # (h, l, c) of the previous bar
        self.atr = self.tr_s = self.pdm_s = self.mdm_s = self.adx = None
        self.pdi = self.mdi = np.nan
        self.ema_f = self.ema_s = None
        self.ema_s_hist = deque(maxlen=c.slope_bars + 1)
        self.closes = deque(maxlen=c.bb_len)
        self.highs = deque(maxlen=c.don_len)
        self.lows = deque(maxlen=c.don_len)
        self.rets = deque(maxlen=c.rv_len)
        self.vol_hist = deque(maxlen=c.vol_lookback)
        self.don_hi = self.don_lo = np.nan     # of the bars BEFORE the current one
        self.slope = self.z = self.mean = self.sd = self.vol_pct = self.rv = np.nan
        self.raw = self.state = None
        self.pending, self.pending_n = None, 0

    @staticmethod
    def _wilder(prev, x, a):
        return x if prev is None else prev + a * (x - prev)

    def update(self, h, l, c):
        cfg = self.c
        # Donchian of the previous don-len bars, captured before this bar joins the window
        if len(self.highs) == cfg.don_len:
            self.don_hi, self.don_lo = max(self.highs), min(self.lows)
        if self.prev is None:
            tr, pdm, mdm = h - l, 0.0, 0.0
        else:
            ph, pl, pc = self.prev
            tr = max(h - l, abs(h - pc), abs(l - pc))
            up, dn = h - ph, pl - l
            pdm = up if up > dn and up > 0 else 0.0
            mdm = dn if dn > up and dn > 0 else 0.0
            if pc > 0 and c > 0:
                self.rets.append(math.log(c / pc))
        self.prev = (h, l, c)
        self.n += 1
        self.atr = self._wilder(self.atr, tr, self.a_atr)
        self.tr_s = self._wilder(self.tr_s, tr, self.a_adx)
        self.pdm_s = self._wilder(self.pdm_s, pdm, self.a_adx)
        self.mdm_s = self._wilder(self.mdm_s, mdm, self.a_adx)
        if self.tr_s > 0:
            self.pdi, self.mdi = 100 * self.pdm_s / self.tr_s, 100 * self.mdm_s / self.tr_s
            s = self.pdi + self.mdi
            dx = 100 * abs(self.pdi - self.mdi) / s if s > 0 else 0.0
            self.adx = self._wilder(self.adx, dx, self.a_adx)
        self.ema_f = self._wilder(self.ema_f, c, self.a_f)
        self.ema_s = self._wilder(self.ema_s, c, self.a_s)
        self.ema_s_hist.append(self.ema_s)
        if len(self.ema_s_hist) == self.ema_s_hist.maxlen and self.atr > 0:
            self.slope = (self.ema_s_hist[-1] - self.ema_s_hist[0]) / (cfg.slope_bars * self.atr)
        self.closes.append(c)
        if len(self.closes) == cfg.bb_len:
            arr = np.fromiter(self.closes, float, len(self.closes))
            self.mean, self.sd = float(arr.mean()), float(arr.std())
            self.z = (c - self.mean) / self.sd if self.sd > 0 else 0.0
        self.highs.append(h)
        self.lows.append(l)
        if len(self.rets) >= max(5, cfg.rv_len // 2):
            self.rv = float(np.std(self.rets)) * math.sqrt(self.bars_per_year)
        v = self.atr / c if cfg.vol_measure == "atr" else self.rv
        if v == v:                                          # not NaN
            hist = self.vol_hist
            if len(hist) >= max(20, cfg.vol_lookback // 2):
                self.vol_pct = sum(1 for x in hist if x <= v) / len(hist)
            hist.append(v)
        self._classify()

    def _classify(self):
        cfg = self.c
        if self.adx is None or self.slope != self.slope or self.vol_pct != self.vol_pct:
            return
        if self.vol_pct >= cfg.vol_high:
            vol = "high"
        elif self.vol_pct <= cfg.vol_low:
            vol = "low"
        else:
            vol = "normal"
        if self.adx >= cfg.adx_trend and abs(self.slope) >= cfg.slope_min:
            d = 1 if self.slope > 0 else -1
            ok = not cfg.di_confirm or (self.pdi > self.mdi) == (d > 0)
            trend = ("up" if d > 0 else "down") if ok else "neutral"
        elif self.adx <= cfg.adx_range:
            trend = "range"
        else:
            trend = "neutral"
        self.raw = f"{trend}/{vol}"
        if self.raw == self.pending:
            self.pending_n += 1
        else:
            self.pending, self.pending_n = self.raw, 1
        if self.pending_n >= cfg.confirm_bars:
            self.state = self.pending

    @property
    def ready(self):
        return self.state is not None and self.n >= self.c.warmup_bars

    @property
    def trend(self):
        return self.state.split("/")[0] if self.state else None

    @property
    def vol(self):
        return self.state.split("/")[1] if self.state else None


# ====================================================================== strategy
class MarketStateFuturesConfig(StrategyConfig, frozen=True):
    venue: str = "GLBX"
    root: str = "CL"
    bar_source: str = "INTERNAL"       # INTERNAL: bars from trades; EXTERNAL: 1-min bars in catalog
    trade_start_date: str = "2026-01-12"
    bar_minutes: int = 15
    # indicators
    atr_len: int = 14
    adx_len: int = 14
    ema_fast: int = 20
    ema_slow: int = 50
    slope_bars: int = 10
    bb_len: int = 40
    don_len: int = 20
    rv_len: int = 32
    vol_measure: str = "atr"           # atr | rv
    vol_lookback: int = 480            # bars in the volatility percentile window (~5 CL days at 15 min)
    warmup_bars: int = 100             # bars per contract before it may be traded
    # state classifier
    adx_trend: float = 25.0
    adx_range: float = 20.0
    slope_min: float = 0.03            # |slow EMA slope| in ATRs per bar for a trend
    di_confirm: int = 1                # +DI/-DI must agree with the slope
    vol_high: float = 0.80             # vol percentile at/above = high
    vol_low: float = 0.20              # vol percentile at/below = low
    confirm_bars: int = 2              # bars a new state must persist before it is active
    # tactics
    tactics: str = "both"              # both | trend | revert
    high_vol: str = "trend"            # tactics allowed to open in high vol: flat|trend|revert|all
    low_vol: str = "all"               # same for low vol
    trend_entry: str = "breakout"      # breakout | pullback | both
    revert_z: float = 2.0
    revert_exit: float = 0.0           # revert target: |z| back to this on the entry side
    stop_atr: float = 2.0
    trail_atr: float = 3.0             # trend trades: chandelier trail (0 = off)
    exit_on_state: int = 1
    max_hold_bars: int = 0             # time stop (0 = off)
    # session (New York time)
    session_open: str = "09:00"
    session_close: str = "14:30"
    trade_start: str = "09:30"
    last_entry: str = "13:45"
    flat_time: str = "14:20"
    overnight: int = 0                 # 1 = hold positions overnight (still flat on Fridays)
    max_trades_day: int = 3
    eia_block: str = "10:20-10:50"     # no new entries on eia_weekday in this window ("" = off)
    eia_weekday: int = 2
    roll_days: int = 8
    # sizing
    contracts: int = 1
    risk_usd: float = 0.0              # > 0: size by risk instead of `contracts`
    max_contracts: int = 10
    output_dir: str = "results_market_state"


class MarketStateFutures(Strategy):

    def __init__(self, config: MarketStateFuturesConfig):
        super().__init__(config)
        self.c = c = config
        checks = dict(tactics=("both", "trend", "revert"), high_vol=GATES, low_vol=GATES,
                      trend_entry=("breakout", "pullback", "both"), vol_measure=("atr", "rv"))
        for k, ok in checks.items():
            if getattr(c, k) not in ok:
                raise ValueError(f"{k} must be one of {ok}, not {getattr(c, k)!r}")
        if c.adx_range > c.adx_trend:
            raise ValueError("adx_range must be <= adx_trend")
        self.venue = Venue(c.venue)
        self.first_day = pd.Timestamp(c.trade_start_date).date()
        self.t_open, self.t_close = _hm(c.session_open), _hm(c.session_close)
        self.t_trade, self.t_last, self.t_flat = _hm(c.trade_start), _hm(c.last_entry), _hm(c.flat_time)
        self.eia = tuple(_hm(x) for x in c.eia_block.split("-")) if c.eia_block else None
        self.ms: dict[str, MarketState] = {}
        self.futs: dict[str, FuturesContract] = {}
        self.cur_fut = None
        self.pos = None
        self.i = 0                                      # bars of the traded contract
        self.trades, self.daily, self.states = [], [], []
        self.day = None
        self.holidays: dict[int, set] = {}
        self.last_bar_ns = 0
        self.start_equity = None

    # ------------------------------------------------------------------ setup
    def on_start(self):
        for i in self.cache.instruments(venue=self.venue):
            if type(i) is FuturesContract and is_root_future(i.raw_symbol.value, self.c.root):
                self.futs[i.raw_symbol.value] = i
                self.ms[i.raw_symbol.value] = MarketState(self.c)
        if not self.futs:
            self.log.error(f"No {self.c.root} futures in the catalog")
            self.stop()
            return
        for f in self.futs.values():
            self.subscribe_bars(self._bar_type(f.id))
        step = timedelta(minutes=5)
        first = pd.Timestamp(self.clock.utc_now()).floor("5min") + step
        self.clock.set_timer("guard", step, start_time=first.to_pydatetime(), callback=self._on_guard)
        self.log.info(f"{self.c.root}: futures {sorted(self.futs)}, {self.c.bar_minutes}-min bars "
                      f"({self.c.bar_source}), tactics {self.c.tactics}, high-vol {self.c.high_vol}")

    def _bar_type(self, iid):
        n = int(self.c.bar_minutes)
        if self.c.bar_source == "EXTERNAL" and n > 1:
            return BarType.from_str(f"{iid}-{n}-MINUTE-LAST-INTERNAL@1-MINUTE-EXTERNAL")
        return BarType.from_str(f"{iid}-{n}-MINUTE-LAST-{self.c.bar_source}")

    # ------------------------------------------------------------------ helpers
    def _ny(self, ns=None):
        ns = self.clock.timestamp_ns() if ns is None else ns
        return pd.Timestamp(ns - ns % 1000, tz="UTC").tz_convert(NY).to_pydatetime().replace(tzinfo=None)

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

    def _cur_ms(self):
        return self.ms[self.cur_fut.raw_symbol.value]

    def _record_day(self, new_date):
        if self.day is not None and self.day["date"] >= self.first_day:
            d, m = self.day, self._cur_ms() if self.cur_fut else None
            row = dict(date=d["date"], equity=self._equity(), contract=d["contract"],
                       state_close=m.state if m else None, adx=m.adx if m else np.nan,
                       vol_pct=m.vol_pct if m else np.nan, atr=m.atr if m else np.nan,
                       trades=d["trades"])
            for k in ("up", "down", "range", "neutral"):
                row[f"bars_{k}"] = d["counts"].get(k, 0)
            self.daily.append(row)
        self.day = None if new_date is None else dict(
            date=new_date, trades=0, counts={},
            contract=self.cur_fut.raw_symbol.value if self.cur_fut else None)

    def _allowed(self, m):
        """Tactics that may open in the current state."""
        trend, vol = m.trend, m.vol
        kinds = {"up": {"trend"}, "down": {"trend"}, "range": {"revert"}}.get(trend, set())
        gate = {"high": self.c.high_vol, "low": self.c.low_vol}.get(vol, "all")
        if gate != "all":
            kinds &= set() if gate == "flat" else {gate}
        if self.c.tactics != "both":
            kinds &= {self.c.tactics}
        return kinds

    def _flat_now(self, now):
        """True when an open position must be closed for the day."""
        if self.c.overnight:                               # hold overnight, but not into the weekend
            return now.weekday() == 4 and now.time() >= self.t_flat
        return now.time() >= self.t_flat or now.time() < self.t_open

    # ------------------------------------------------------------------ guard timer
    def _on_guard(self, _event):
        """Flat-time and halt exits that must not depend on bars arriving."""
        if not self.pos or self.pos.get("reason"):
            return
        now = self._ny()
        in_session = self.t_open <= now.time() < self.t_flat
        quiet = self.clock.timestamp_ns() - self.last_bar_ns > (self.c.bar_minutes + 15) * NS_MIN
        if self._flat_now(now):
            self._exit("eod")
        elif in_session and quiet:
            self._exit("halt")

    # ------------------------------------------------------------------ bars
    def on_bar(self, bar: Bar):
        if float(bar.volume) <= 0:
            return
        sym = bar.bar_type.instrument_id.symbol.value
        m = self.ms.get(sym)
        if m is None:
            return
        h, l, c = float(bar.high), float(bar.low), float(bar.close)
        if c <= 0:
            return
        prev_state = m.state
        m.update(h, l, c)                               # every contract, every bar

        tf = self._trade_future()
        if tf is None or bar.bar_type.instrument_id != tf.id:
            return
        if self.cur_fut is None or tf.id != self.cur_fut.id:
            if self.pos and not self.pos.get("reason"):
                self._exit("roll")
            self.cur_fut = tf
            self.log.info(f"trading {sym} ({m.n} bars of history, state {m.state})")
        self.last_bar_ns = bar.ts_event
        self.i += 1
        t = self._ny(bar.ts_event)
        if t.date() >= self.first_day:
            self.states.append(dict(
                time=t, contract=sym, close=c, atr=m.atr, adx=m.adx, pdi=m.pdi, mdi=m.mdi,
                slope=m.slope, z=m.z, ema_fast=m.ema_f, ema_slow=m.ema_s, don_hi=m.don_hi,
                don_lo=m.don_lo, vol_pct=m.vol_pct, rv=m.rv, raw=m.raw, state=m.state,
                position=(self.pos["side"] * self.pos["n"]) if self.pos and "entry_px" in self.pos else 0))

        if self.pos and not self.pos.get("reason"):
            self._manage(m, h, l, c, state_changed=m.state != prev_state)

        if not (self.t_open < t.time() <= self.t_close):
            return                                       # overnight: indicators + management only
        if self.day is None or self.day["date"] != t.date():
            self._record_day(t.date())
        if t.date() < self.first_day:
            return
        if self.start_equity is None:
            self.start_equity = self._equity()
        if m.trend:
            self.day["counts"][m.trend] = self.day["counts"].get(m.trend, 0) + 1

        if self._flat_now(t):
            if self.pos and not self.pos.get("reason"):
                self._exit("eod")
            return
        if self.pos:
            return
        if t.year not in self.holidays:
            self.holidays[t.year] = nyse_holidays(t.year)
        if t.time() < self.t_trade or t.time() > self.t_last or t.date() in self.holidays[t.year] \
                or self.day["trades"] >= self.c.max_trades_day or not m.ready \
                or (self.eia is not None and t.weekday() == self.c.eia_weekday
                    and self.eia[0] <= t.time() <= self.eia[1]):
            return
        self._signal(m, h, l, c)

    def _signal(self, m, h, l, c):
        cfg, kinds = self.c, self._allowed(m)
        if "trend" in kinds:
            side = 1 if m.trend == "up" else -1
            go = False
            if cfg.trend_entry in ("breakout", "both") and m.don_hi == m.don_hi:
                go = (side > 0 and c > m.don_hi) or (side < 0 and c < m.don_lo)
            if not go and cfg.trend_entry in ("pullback", "both"):
                if side > 0:
                    go = l <= m.ema_f < c and m.ema_f > m.ema_s
                else:
                    go = h >= m.ema_f > c and m.ema_f < m.ema_s
            if go:
                return self._enter(side, "trend", m, c)
        if "revert" in kinds and m.z == m.z:
            if m.z <= -cfg.revert_z:
                return self._enter(+1, "revert", m, c)
            if m.z >= cfg.revert_z:
                return self._enter(-1, "revert", m, c)

    # ------------------------------------------------------------------ orders
    def _snap(self, px):
        inc = float(self.cur_fut.price_increment)
        return round(px / inc) * inc

    def _size(self, risk_pts):
        if self.c.risk_usd <= 0:
            return int(self.c.contracts)
        per = risk_pts * float(self.cur_fut.multiplier)
        n = int(self.c.risk_usd // per) if per > 0 else 0
        return max(0, min(n, int(self.c.max_contracts)))

    def _enter(self, side, kind, m, px):
        dist = self.c.stop_atr * m.atr
        n = self._size(dist)
        if n <= 0 or dist <= 0:
            return
        self.pos = dict(side=side, kind=kind, iid=self.cur_fut.id, n=n, stop=self._snap(px - side * dist),
                        best=px, i0=self.i, opened=self._ny(), cash=0.0, state=m.state,
                        adx=m.adx, vol_pct=m.vol_pct, z=m.z, atr=m.atr)
        self.pos["stop0"] = self.pos["stop"]
        self.day["trades"] += 1
        self.submit_order(self.order_factory.market(
            self.cur_fut.id, OrderSide.BUY if side > 0 else OrderSide.SELL,
            self.cur_fut.make_qty(n), tags=[f"open:{kind}"]))

    def _place_stop(self):
        """Resting reduce-only stop at the venue once the entry is filled."""
        p = self.pos
        inst = self.cache.instrument(p["iid"])
        order = self.order_factory.stop_market(
            p["iid"], OrderSide.SELL if p["side"] > 0 else OrderSide.BUY, inst.make_qty(p["n"]),
            trigger_price=inst.make_price(p["stop"]), reduce_only=True, tags=["close:stop"])
        p["stop_id"] = order.client_order_id
        self.submit_order(order)

    def _move_stop(self, px):
        p = self.pos
        o = self.cache.order(p["stop_id"]) if p.get("stop_id") else None
        p["stop"] = px
        if o is None or not o.is_open:
            return
        self.modify_order(o, trigger_price=self.cache.instrument(p["iid"]).make_price(px))

    def _cancel_stop(self):
        sid = self.pos.get("stop_id")
        o = self.cache.order(sid) if sid is not None else None
        if o is not None and o.is_open:
            self.cancel_order(o)

    def on_order_rejected(self, e: OrderRejected):
        p = self.pos
        if p and e.client_order_id == p.get("stop_id"):  # stop already through the market
            p["stop_id"] = None
            if not p.get("reason"):
                self._exit("stop")

    def _exit(self, reason):
        p = self.pos
        p["reason"] = reason
        self._cancel_stop()
        q = int(self.portfolio.net_position(p["iid"]))
        if q == 0:
            if "entry_px" not in p:                       # never filled
                self.pos = None
            return
        inst = self.cache.instrument(p["iid"])
        self.submit_order(self.order_factory.market(
            p["iid"], OrderSide.SELL if q > 0 else OrderSide.BUY, inst.make_qty(abs(q)),
            tags=[f"close:{reason}"]))

    def _manage(self, m, h, l, c, state_changed):
        p = self.pos
        if p["iid"] != self.cur_fut.id or int(self.portfolio.net_position(p["iid"])) == 0:
            return                                       # entry not filled yet
        side, cfg = p["side"], self.c
        p["exit_state"] = m.state
        if p.get("stop_id") is None and ((side > 0 and l <= p["stop"]) or (side < 0 and h >= p["stop"])):
            return self._exit("stop")                    # only if the resting stop could not be placed
        if p["kind"] == "trend":
            if cfg.trail_atr > 0:
                p["best"] = max(p["best"], c) if side > 0 else min(p["best"], c)
                new = self._snap(p["best"] - side * cfg.trail_atr * m.atr)
                if (side > 0 and new > p["stop"]) or (side < 0 and new < p["stop"]):
                    if (side > 0 and new >= c) or (side < 0 and new <= c):
                        return self._exit("trail")
                    self._move_stop(new)
            if cfg.exit_on_state and state_changed and m.state:
                against = m.trend == "range" or m.trend == ("down" if side > 0 else "up")
                if against or ("trend" not in self._allowed(m) and m.trend in ("up", "down")):
                    return self._exit("state")
        else:
            if m.z == m.z and ((side > 0 and m.z >= -cfg.revert_exit) or (side < 0 and m.z <= cfg.revert_exit)):
                return self._exit("target")
            if cfg.exit_on_state and state_changed and m.state:
                against = m.trend == ("down" if side > 0 else "up")
                if against or ("revert" not in self._allowed(m) and m.trend == "range"):
                    return self._exit("state")
        if cfg.max_hold_bars > 0 and self.i - p["i0"] >= cfg.max_hold_bars:
            self._exit("time")

    def on_order_filled(self, e: OrderFilled):
        p = self.pos
        if p is None or e.instrument_id != p["iid"]:
            return
        signed = float(e.last_qty) * (1 if e.order_side == OrderSide.BUY else -1)
        p["cash"] -= signed * float(e.last_px)
        p.setdefault("entry_px", float(e.last_px))
        if e.client_order_id == p.get("stop_id") and not p.get("reason"):
            p["reason"] = "trail" if p["stop"] != p["stop0"] else "stop"
        net = int(self.portfolio.net_position(p["iid"]))
        if not p.get("reason") and "stop_id" not in p and abs(net) == p["n"]:
            self._place_stop()
        if p.get("reason") and net == 0:
            pnl = p["cash"] * float(self.cache.instrument(p["iid"]).multiplier)
            self.trades.append(dict(
                kind=p["kind"], side="long" if p["side"] > 0 else "short", opened=p["opened"],
                closed=self._ny(), reason=p["reason"], n=p["n"], entry=p["entry_px"],
                exit=float(e.last_px), stop_init=p["stop0"], stop_final=p["stop"],
                entry_state=p["state"], exit_state=p.get("exit_state", p["state"]),
                adx=p["adx"], vol_pct=p["vol_pct"], z=p["z"], atr=p["atr"],
                bars=self.i - p["i0"], pnl=pnl, contract=str(p["iid"])))
            self.log.info(f"{p['kind']} {'L' if p['side'] > 0 else 'S'} [{p['state']}] "
                          f"{p['opened']:%Y-%m-%d %H:%M} -> {self._ny():%m-%d %H:%M} {p['reason']:6s} "
                          f"n={p['n']} pnl=${pnl:,.0f}")
            self.pos = None

    def on_stop(self):
        self._record_day(None)
        t = pd.DataFrame(self.trades)
        st = pd.DataFrame(self.states)
        eq = self._equity()
        stats = {"Total P&L (incl. fees)": f"{eq - float(self.start_equity or eq):,.0f}",
                 "Trades": len(t),
                 "Win rate": f"{(t['pnl'] > 0).mean():.0%}" if len(t) else "n/a",
                 "Avg trade P&L": f"{t['pnl'].mean():,.0f}" if len(t) else "n/a",
                 "Long / short": f"{(t['side'] == 'long').sum()} / {(t['side'] == 'short').sum()}" if len(t) else "0 / 0"}
        if len(st) and st["state"].notna().any():
            share = st["state"].value_counts(normalize=True)
            for s, v in share.items():
                stats[f"Time in {s}"] = f"{v:.0%}"
        if len(t):
            for k, g in t.groupby("kind"):
                stats[f"Kind {k}"] = f"{len(g)} trades, {g['pnl'].sum():,.0f}, win {(g['pnl'] > 0).mean():.0%}"
            for s, g in t.groupby("entry_state"):
                stats[f"Entry state {s}"] = f"{len(g)} trades, {g['pnl'].sum():,.0f}"
            for r, g in t.groupby("reason"):
                stats[f"Exit {r}"] = f"{len(g)} trades, {g['pnl'].sum():,.0f}"
        for k, v in stats.items():
            self.log.info(f"{k}: {v}")
        os.makedirs(self.c.output_dir, exist_ok=True)
        t.to_csv(os.path.join(self.c.output_dir, "trades.csv"), index=False)
        pd.DataFrame(self.daily).to_csv(os.path.join(self.c.output_dir, "signals.csv"), index=False)
        st.to_csv(os.path.join(self.c.output_dir, "states.csv"), index=False)
        pd.Series(stats).to_csv(os.path.join(self.c.output_dir, "summary.csv"), header=False)


# ====================================================================== runner
FIELDS = list(MarketStateFuturesConfig.__struct_fields__)
DEFAULTS = dict(zip(FIELDS, MarketStateFuturesConfig.__struct_defaults__))


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


def default_warmup_days(params):
    """Calendar days of data before --start so the slowest indicator is full on day one."""
    p = {**DEFAULTS, **params}
    bars = max(p["vol_lookback"], 3 * p["adx_len"], 3 * p["ema_slow"], p["bb_len"], p["don_len"],
               p["warmup_bars"])
    trading_days = bars * p["bar_minutes"] / (22 * 60)      # CL/ES trade ~23 h a day
    return int(math.ceil(trading_days * 7 / 5)) + 3


def build_config(catalog_path, start, end, cash, fee, params, out_dir, venue="GLBX",
                 oi_path=None, chunk_size=None, log_level="INFO", bar_source="auto",
                 warmup_days=None, quotes=False, **_ignored):
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
    if warmup_days is None:
        warmup_days = default_warmup_days(params)
    data_start = (pd.Timestamp(start) - pd.Timedelta(days=warmup_days)).isoformat()
    data_end = (pd.Timestamp(end) + pd.Timedelta(days=1)).isoformat()
    ids = [str(f.id) for f in futs]
    strategy = ImportableStrategyConfig(
        strategy_path="market_state_futures:MarketStateFutures",
        config_path="market_state_futures:MarketStateFuturesConfig",
        config=dict(params, venue=venue, bar_source=bar_source,
                    trade_start_date=str(pd.Timestamp(start).date()), output_dir=out_dir))
    venue_cfg = BacktestVenueConfig(
        name=venue, oms_type="NETTING", account_type="MARGIN", base_currency="USD",
        starting_balances=[f"{cash:.2f} USD"],
        fee_model=ImportableFeeModelConfig(
            fee_model_path="nautilus_trader.backtest.models:PerContractFeeModel",
            config_path="nautilus_trader.backtest.config:PerContractFeeModelConfig",
            config={"commission": f"{fee:.2f} USD"}))
    # Fills from trade ticks by default: the catalog's quotes are bbo-1m samples, which go stale
    # inside the minute and can trigger resting stops on a quote the market had already left.
    data = [BacktestDataConfig(catalog_path=catalog_path, data_cls=QuoteTick, instrument_ids=ids,
                               start_time=data_start, end_time=data_end)] if quotes else []
    if bar_source == "EXTERNAL":
        data.append(BacktestDataConfig(catalog_path=catalog_path, data_cls=Bar,
                                       bar_types=[f"{i}-1-MINUTE-LAST-EXTERNAL" for i in ids],
                                       start_time=data_start, end_time=data_end))
    if bar_source != "EXTERNAL" or not quotes:   # trades build the bars and/or drive the fills
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
    ap.add_argument("--start", default="2026-01-12",
                    help="first trading day; earlier data (--warmup-days) only warms the indicators")
    ap.add_argument("--end", default="2026-03-31")
    ap.add_argument("--warmup-days", type=int, default=None,
                    help="calendar days of data before --start (default: from the indicator lengths)")
    ap.add_argument("--cash", type=float, default=1_000_000.0)
    ap.add_argument("--fee", type=float, default=2.0, help="commission per contract, USD")
    ap.add_argument("--param", action="append", help="key=value (dashes ok), repeatable")
    ap.add_argument("--params-json", default=None)
    ap.add_argument("--venue", default="GLBX")
    ap.add_argument("--bars", default="auto", choices=["auto", "EXTERNAL", "INTERNAL"])
    ap.add_argument("--quotes", action="store_true",
                    help="also load QuoteTicks for the matching engine (default: fills from trades only)")
    ap.add_argument("--chunk-size", type=int, default=250_000,
                    help="records loaded per batch; lower = less memory (default 250,000)")
    ap.add_argument("--out", default="results_market_state")
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args()

    from nautilus_trader.backtest.node import BacktestNode

    params, _ = parse_params(a.param, a.params_json)
    os.makedirs(a.out, exist_ok=True)
    cfg, info = build_config(a.catalog, a.start, a.end, a.cash, a.fee, params, a.out, venue=a.venue,
                             chunk_size=a.chunk_size, log_level=a.log_level, bar_source=a.bars,
                             warmup_days=a.warmup_days, quotes=a.quotes)
    print(f"futures {info['futures']}, bars {info['bar_source']}, data from {info['data_start'][:10]}, "
          f"fills from {'quotes + trades' if a.quotes else 'trades'}")
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
