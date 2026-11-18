"""Low-drawdown CL futures strategy — standalone NautilusTrader script.

Goal: a small, steady equity curve rather than big wins. The edge is deliberately modest
(fading stretched moves back to the session VWAP on quiet days); most of the script is the
risk layer that keeps losses small and stops trading when things go wrong.

Signals (entry vwap | pullback | both), on N-minute bars of the front CL future:
  vwap      Session VWAP anchored at session-open (09:00 ET) with volume-weighted SD bands.
            Only in a quiet regime: ADX <= adx-max, |slow EMA slope| <= slope-max (ATRs per
            bar) and volatility percentile <= vol-max.
            Long when a bar trades below VWAP - band-k x SD and (confirm 1) closes back
            above that band: a rejected push, not a falling knife. Short mirrored.
            Needs reward-to-risk >= min-rr, reward = distance to VWAP, risk = the stop.
            Target: close back at VWAP (target-sd x SD short of it, 0 = VWAP itself).
  pullback  Trend continuation: ADX >= adx-trend, slope >= slope-min with price on the
            trend side of VWAP; the bar dips to the fast EMA and closes back on the trend
            side. Target tp2-r x R.
Risk per trade (R) = stop-atr x ATR from entry, resting reduce-only stop at the venue.

Risk layer:
  sizing      contracts = floor(risk-pct x equity x dd-scale / (stop distance x multiplier)),
              capped at max-contracts; 0 = trade skipped (stop too wide for the budget).
              risk-pct 0 = fixed `contracts`.
  partial     with >= 2 contracts, tp1-frac of them are taken at tp1-r x R by a resting
              limit, then the stop on the rest moves to breakeven.
  breakeven   without a partial, the stop moves to entry once the trade has been be-r x R
              in profit (0 = off).
  time stop   max-hold-bars.
  day limit   no new entries after day-loss-pct of the day's starting equity is lost (open
              trades are closed too) or after max-losses-day losing trades.
  governor    drawdown from the equity high-water mark: full size below dd-soft, size scaled
              linearly down to dd-min-scale at dd-halt; at dd-halt no entries for halt-days
              trading days, then the high-water mark is re-based to current equity and the
              strategy restarts at dd-min-scale size.
Flat every day by flat-time; no entries around the Wednesday EIA report or on holidays.

Outputs in --out: trades.csv (with R multiples), signals.csv (daily equity, drawdown, size
scale), summary.csv (P&L, return, max drawdown intraday and daily, Sharpe, Sortino, profit
factor, worst day ...). Use a realistic --cash (default 100,000) because size depends on it.

  python lowdd_cl_futures.py --catalog fop_data/catalog --start 2026-01-12 --end 2026-03-31 --out results_lowdd
  python lowdd_cl_futures.py --catalog fop_data/catalog --param entry=both --param risk-pct=0.003 ...
  python optimize.py --module lowdd_cl_futures --catalog fop_data/catalog --grid lowdd_grid.json \
      --cash 100000 --target calmar --start 2026-01-12 --end 2026-03-31 --out opt_lowdd
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


# ====================================================================== indicators
class Indicators:
    """ATR, ADX, EMAs + slope, volatility percentile and session VWAP for one contract."""

    def __init__(self, c, t_anchor):
        self.c = c
        self.t_anchor = t_anchor
        self.a_atr = 1.0 / c.atr_len
        self.a_adx = 1.0 / c.adx_len
        self.a_f = 2.0 / (c.ema_fast + 1)
        self.a_s = 2.0 / (c.ema_slow + 1)
        self.n = 0
        self.prev = None
        self.atr = self.tr_s = self.pdm_s = self.mdm_s = self.adx = None
        self.ema_f = self.ema_s = None
        self.ema_s_hist = deque(maxlen=c.slope_bars + 1)
        self.vol_hist = deque(maxlen=c.vol_lookback)
        self.slope = self.vol_pct = np.nan
        # session VWAP
        self.anchor_date = None
        self.sv = self.svp = self.svp2 = 0.0
        self.vwap = self.vsd = np.nan
        self.vbars = 0

    @staticmethod
    def _wilder(prev, x, a):
        return x if prev is None else prev + a * (x - prev)

    def update(self, t, h, l, c, v):
        cfg = self.c
        if self.prev is None:
            tr, pdm, mdm = h - l, 0.0, 0.0
        else:
            ph, pl, pc = self.prev
            tr = max(h - l, abs(h - pc), abs(l - pc))
            up, dn = h - ph, pl - l
            pdm = up if up > dn and up > 0 else 0.0
            mdm = dn if dn > up and dn > 0 else 0.0
        self.prev = (h, l, c)
        self.n += 1
        self.atr = self._wilder(self.atr, tr, self.a_atr)
        self.tr_s = self._wilder(self.tr_s, tr, self.a_adx)
        self.pdm_s = self._wilder(self.pdm_s, pdm, self.a_adx)
        self.mdm_s = self._wilder(self.mdm_s, mdm, self.a_adx)
        if self.tr_s > 0:
            pdi, mdi = 100 * self.pdm_s / self.tr_s, 100 * self.mdm_s / self.tr_s
            s = pdi + mdi
            dx = 100 * abs(pdi - mdi) / s if s > 0 else 0.0
            self.adx = self._wilder(self.adx, dx, self.a_adx)
        self.ema_f = self._wilder(self.ema_f, c, self.a_f)
        self.ema_s = self._wilder(self.ema_s, c, self.a_s)
        self.ema_s_hist.append(self.ema_s)
        if len(self.ema_s_hist) == self.ema_s_hist.maxlen and self.atr > 0:
            self.slope = (self.ema_s_hist[-1] - self.ema_s_hist[0]) / (cfg.slope_bars * self.atr)
        natr = self.atr / c
        hist = self.vol_hist
        if len(hist) >= max(20, cfg.vol_lookback // 2):
            self.vol_pct = sum(1 for x in hist if x <= natr) / len(hist)
        hist.append(natr)
        # VWAP: reset on the first bar that closes after the anchor time each day
        if t.time() > self.t_anchor and self.anchor_date != t.date():
            self.anchor_date = t.date()
            self.sv = self.svp = self.svp2 = 0.0
            self.vbars = 0
        if self.anchor_date == t.date() and v > 0:
            tp = (h + l + c) / 3.0
            self.sv += v
            self.svp += v * tp
            self.svp2 += v * tp * tp
            self.vbars += 1
            self.vwap = self.svp / self.sv
            self.vsd = math.sqrt(max(self.svp2 / self.sv - self.vwap ** 2, 0.0))
        elif self.anchor_date != t.date():
            self.vwap = self.vsd = np.nan

    @property
    def ready(self):
        return (self.n >= self.c.warmup_bars and self.adx is not None and self.slope == self.slope
                and self.vol_pct == self.vol_pct)


# ====================================================================== strategy
class LowDdClFuturesConfig(StrategyConfig, frozen=True):
    venue: str = "GLBX"
    root: str = "CL"
    bar_source: str = "INTERNAL"       # INTERNAL: bars from trades; EXTERNAL: 1-min bars in catalog
    trade_start_date: str = "2026-01-12"
    bar_minutes: int = 5
    # indicators
    atr_len: int = 14
    adx_len: int = 14
    ema_fast: int = 20
    ema_slow: int = 60
    slope_bars: int = 12
    vol_lookback: int = 1380           # bars in the volatility percentile window (~5 CL days at 5 min)
    warmup_bars: int = 300
    # signals
    entry: str = "vwap"                # vwap | pullback | both
    band_k: float = 2.0                # VWAP band in volume-weighted SDs
    confirm: int = 1                   # 1 = bar must close back inside the band
    min_band_bars: int = 6             # bars since the VWAP anchor before vwap entries
    min_rr: float = 1.0                # vwap: distance to VWAP / stop distance
    target_sd: float = 0.0             # vwap: exit this many SDs short of VWAP
    adx_max: float = 25.0              # vwap regime: ADX at/below
    slope_max: float = 0.04            # vwap regime: |slow EMA slope| (ATR/bar) at/below
    vol_max: float = 0.85              # both: no entries above this volatility percentile
    adx_trend: float = 25.0            # pullback regime
    slope_min: float = 0.04
    tp2_r: float = 2.0                 # pullback: final target in R
    # trade management
    stop_atr: float = 1.5
    tp1_r: float = 1.0                 # partial take profit at this R (0 = off)
    tp1_frac: float = 0.5
    be_r: float = 1.0                  # move stop to breakeven at this R (0 = off)
    max_hold_bars: int = 24
    # risk layer
    risk_pct: float = 0.004            # equity fraction risked per trade (0 = fixed `contracts`)
    contracts: int = 1
    max_contracts: int = 6
    day_loss_pct: float = 0.008        # stop for the day after losing this fraction (0 = off)
    max_losses_day: int = 2
    dd_soft: float = 0.02              # drawdown where de-risking starts
    dd_halt: float = 0.05              # drawdown where entries halt
    dd_min_scale: float = 0.33
    halt_days: int = 3
    # session (New York time)
    session_open: str = "09:00"        # also the VWAP anchor
    session_close: str = "14:30"
    trade_start: str = "09:35"
    last_entry: str = "13:30"
    flat_time: str = "14:20"
    max_trades_day: int = 3
    eia_block: str = "10:15-11:00"
    eia_weekday: int = 2
    roll_days: int = 8
    output_dir: str = "results_lowdd"


class LowDdClFutures(Strategy):

    def __init__(self, config: LowDdClFuturesConfig):
        super().__init__(config)
        self.c = c = config
        if c.entry not in ("vwap", "pullback", "both"):
            raise ValueError(f"entry must be vwap | pullback | both, not {c.entry!r}")
        if c.dd_halt <= c.dd_soft:
            raise ValueError("dd_halt must be > dd_soft")
        self.venue = Venue(c.venue)
        self.first_day = pd.Timestamp(c.trade_start_date).date()
        self.t_open, self.t_close = _hm(c.session_open), _hm(c.session_close)
        self.t_trade, self.t_last, self.t_flat = _hm(c.trade_start), _hm(c.last_entry), _hm(c.flat_time)
        self.eia = tuple(_hm(x) for x in c.eia_block.split("-")) if c.eia_block else None
        self.ind: dict[str, Indicators] = {}
        self.futs: dict[str, FuturesContract] = {}
        self.cur_fut = None
        self.pos = None
        self.i = 0
        self.trades, self.daily = [], []
        self.day = None
        self.holidays: dict[int, set] = {}
        self.last_bar_ns = 0
        self.start_equity = None
        # risk state
        self.hwm = None                 # governor high-water mark (re-based after a halt)
        self.halt_left = 0              # trading days left in a drawdown halt
        self.halts = 0
        self.peak = None                # true running peak, for reporting
        self.max_dd_usd = self.max_dd_pct = 0.0
        self.restart_small = False      # after a halt: trade at dd-min-scale until back above the new HWM

    # ------------------------------------------------------------------ setup
    def on_start(self):
        for i in self.cache.instruments(venue=self.venue):
            if type(i) is FuturesContract and is_root_future(i.raw_symbol.value, self.c.root):
                self.futs[i.raw_symbol.value] = i
                self.ind[i.raw_symbol.value] = Indicators(self.c, self.t_open)
        if not self.futs:
            self.log.error(f"No {self.c.root} futures in the catalog")
            self.stop()
            return
        for f in self.futs.values():
            self.subscribe_bars(self._bar_type(f.id))
        step = timedelta(minutes=5)
        first = pd.Timestamp(self.clock.utc_now()).floor("5min") + step
        self.clock.set_timer("guard", step, start_time=first.to_pydatetime(), callback=self._on_guard)
        self.log.info(f"{self.c.root}: futures {sorted(self.futs)}, {self.c.bar_minutes}-min bars, "
                      f"entry {self.c.entry}, risk {self.c.risk_pct:.2%}/trade")

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

    def _track_equity(self):
        eq = self._equity()
        if eq != eq:
            return eq
        if self.peak is None or eq > self.peak:
            self.peak = eq
        if self.hwm is None or eq > self.hwm:
            self.hwm = eq
        dd = self.peak - eq
        if dd > self.max_dd_usd:
            self.max_dd_usd = dd
        self.max_dd_pct = max(self.max_dd_pct, dd / self.peak if self.peak > 0 else 0.0)
        return eq

    def _dd_scale(self, eq):
        """Size multiplier from the governor's drawdown (0 = halted)."""
        c = self.c
        if self.halt_left > 0:
            return 0.0
        dd = 1.0 - eq / self.hwm if self.hwm else 0.0
        if dd < c.dd_soft:
            return 1.0
        if dd >= c.dd_halt:
            return 0.0
        frac = (dd - c.dd_soft) / (c.dd_halt - c.dd_soft)
        return 1.0 - frac * (1.0 - c.dd_min_scale)

    def _new_day(self, date):
        """Close out yesterday's row, roll the governor, open today's."""
        eq = self._equity()
        if self.day is not None and self.day["date"] >= self.first_day:
            d = self.day
            self.daily.append(dict(
                date=d["date"], equity=eq, day_pnl=eq - d["eq0"], contract=d["contract"],
                trades=d["trades"], losses=d["losses"], scale=d["scale"], halted=d["halted"],
                stopped=d["stopped"], dd_from_peak=(self.peak - eq) / self.peak if self.peak else 0.0))
            if self.halt_left > 0:
                self.halt_left -= 1
                if self.halt_left == 0:                   # restart: re-base the high-water mark
                    self.hwm = eq
                    self.restart_small = True
        if date is None:
            self.day = None
            return
        if self.hwm and self.halt_left == 0 and eq <= self.hwm * (1.0 - self.c.dd_halt) \
                and date >= self.first_day:
            self.halt_left = int(self.c.halt_days)
            self.halts += 1
            self.log.warning(f"{date}: drawdown {1 - eq / self.hwm:.1%} >= dd-halt, no entries "
                             f"for {self.c.halt_days} days")
        scale = self._dd_scale(eq)
        if self.restart_small and self.halt_left == 0:
            scale = min(scale, self.c.dd_min_scale)
            if eq >= self.hwm * (1.0 + self.c.dd_soft):   # earned its way back to full size
                self.restart_small = False
        self.day = dict(date=date, eq0=eq, trades=0, losses=0, scale=scale, halted=self.halt_left > 0,
                        stopped=False, contract=self.cur_fut.raw_symbol.value if self.cur_fut else None)

    # ------------------------------------------------------------------ guard timer
    def _on_guard(self, _event):
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
        v = float(bar.volume)
        if v <= 0:
            return
        sym = bar.bar_type.instrument_id.symbol.value
        m = self.ind.get(sym)
        if m is None:
            return
        h, l, c = float(bar.high), float(bar.low), float(bar.close)
        if c <= 0:
            return
        t = self._ny(bar.ts_event)
        m.update(t, h, l, c, v)

        tf = self._trade_future()
        if tf is None or bar.bar_type.instrument_id != tf.id:
            return
        if self.cur_fut is None or tf.id != self.cur_fut.id:
            if self.pos and not self.pos.get("reason"):
                self._exit("roll")
            self.cur_fut = tf
            self.log.info(f"trading {sym} ({m.n} bars of history)")
        self.last_bar_ns = bar.ts_event
        self.i += 1
        if t.date() >= self.first_day:
            eq = self._track_equity()
        if self.pos and not self.pos.get("reason"):
            self._manage(m, h, l, c)

        if not (self.t_open < t.time() <= self.t_close):
            return
        if self.day is None or self.day["date"] != t.date():
            self._new_day(t.date())
        if t.date() < self.first_day:
            return
        if self.start_equity is None:
            self.start_equity = self._equity()
        d = self.day
        # daily loss limit (realised + open)
        if self.c.day_loss_pct > 0 and not d["stopped"] and eq - d["eq0"] <= -self.c.day_loss_pct * d["eq0"]:
            d["stopped"] = True
            self.log.info(f"{t:%Y-%m-%d %H:%M} day loss limit hit ({eq - d['eq0']:,.0f})")
            if self.pos and not self.pos.get("reason"):
                self._exit("daystop")
        if t.time() >= self.t_flat:
            if self.pos and not self.pos.get("reason"):
                self._exit("eod")
            return
        if self.pos or d["stopped"] or d["scale"] <= 0:
            return
        if t.year not in self.holidays:
            self.holidays[t.year] = nyse_holidays(t.year)
        if t.time() < self.t_trade or t.time() > self.t_last or t.date() in self.holidays[t.year] \
                or d["trades"] >= self.c.max_trades_day or d["losses"] >= self.c.max_losses_day \
                or not m.ready or (self.eia is not None and t.weekday() == self.c.eia_weekday
                                   and self.eia[0] <= t.time() <= self.eia[1]):
            return
        self._signal(m, h, l, c)

    def _signal(self, m, h, l, c):
        cfg = self.c
        if m.vol_pct > cfg.vol_max or m.atr <= 0:
            return
        stop_d = cfg.stop_atr * m.atr
        if cfg.entry in ("vwap", "both") and m.vwap == m.vwap and m.vsd > 0 \
                and m.vbars >= cfg.min_band_bars \
                and m.adx <= cfg.adx_max and abs(m.slope) <= cfg.slope_max:
            lo, hi = m.vwap - cfg.band_k * m.vsd, m.vwap + cfg.band_k * m.vsd
            tgt_l, tgt_s = m.vwap - cfg.target_sd * m.vsd, m.vwap + cfg.target_sd * m.vsd
            long_ok = (l <= lo < c) if cfg.confirm else c <= lo
            short_ok = (h >= hi > c) if cfg.confirm else c >= hi
            if long_ok and (tgt_l - c) >= cfg.min_rr * stop_d:
                return self._enter(+1, "vwap", m, c, stop_d)
            if short_ok and (c - tgt_s) >= cfg.min_rr * stop_d:
                return self._enter(-1, "vwap", m, c, stop_d)
        if cfg.entry in ("pullback", "both") and m.adx >= cfg.adx_trend and m.vwap == m.vwap:
            if m.slope >= cfg.slope_min and c > m.vwap and l <= m.ema_f < c and m.ema_f > m.ema_s:
                return self._enter(+1, "pullback", m, c, stop_d)
            if m.slope <= -cfg.slope_min and c < m.vwap and h >= m.ema_f > c and m.ema_f < m.ema_s:
                return self._enter(-1, "pullback", m, c, stop_d)

    # ------------------------------------------------------------------ orders
    def _snap(self, px):
        inc = float(self.cur_fut.price_increment)
        return round(px / inc) * inc

    def _size(self, risk_pts):
        c = self.c
        if c.risk_pct <= 0:
            return int(c.contracts)
        per = risk_pts * float(self.cur_fut.multiplier)
        budget = c.risk_pct * self._equity() * self.day["scale"]
        n = int(budget // per) if per > 0 else 0
        return max(0, min(n, int(c.max_contracts)))

    def _enter(self, side, kind, m, px, stop_d):
        n = self._size(stop_d)
        if n <= 0:
            return
        mult = float(self.cur_fut.multiplier)
        self.pos = dict(side=side, kind=kind, iid=self.cur_fut.id, n=n, stop_d=stop_d,
                        stop=self._snap(px - side * stop_d), i0=self.i, opened=self._ny(), cash=0.0,
                        risk_usd=n * stop_d * mult, adx=m.adx, slope=m.slope, vol_pct=m.vol_pct,
                        vwap=m.vwap, vsd=m.vsd, atr=m.atr, scale=self.day["scale"], partial=False,
                        be=False, mfe=0.0)
        self.day["trades"] += 1
        self.submit_order(self.order_factory.market(
            self.cur_fut.id, OrderSide.BUY if side > 0 else OrderSide.SELL,
            self.cur_fut.make_qty(n), tags=[f"open:{kind}"]))

    def _place_stop(self, qty):
        p = self.pos
        inst = self.cache.instrument(p["iid"])
        order = self.order_factory.stop_market(
            p["iid"], OrderSide.SELL if p["side"] > 0 else OrderSide.BUY, inst.make_qty(qty),
            trigger_price=inst.make_price(p["stop"]), reduce_only=True, tags=["close:stop"])
        p["stop_id"] = order.client_order_id
        self.submit_order(order)

    def _place_tp1(self):
        p, c = self.pos, self.c
        q = int(math.floor(p["n"] * c.tp1_frac))
        if c.tp1_r <= 0 or q < 1 or q >= p["n"]:
            return
        inst = self.cache.instrument(p["iid"])
        px = self._snap(p["entry_px"] + p["side"] * c.tp1_r * p["stop_d"])
        order = self.order_factory.limit(
            p["iid"], OrderSide.SELL if p["side"] > 0 else OrderSide.BUY, inst.make_qty(q),
            price=inst.make_price(px), reduce_only=True, tags=["close:tp1"])
        p["tp1_id"] = order.client_order_id
        self.submit_order(order)

    def _cancel(self, key):
        oid = self.pos.get(key) if self.pos else None
        o = self.cache.order(oid) if oid is not None else None
        if o is not None and o.is_open:
            self.cancel_order(o)

    def _move_stop(self, px):
        p = self.pos
        o = self.cache.order(p["stop_id"]) if p.get("stop_id") else None
        p["stop"] = px
        if o is not None and o.is_open:
            self.modify_order(o, trigger_price=self.cache.instrument(p["iid"]).make_price(px))

    def on_order_rejected(self, e: OrderRejected):
        p = self.pos
        if p and e.client_order_id == p.get("stop_id"):  # stop already through the market
            p["stop_id"] = None
            if not p.get("reason"):
                self._exit("stop")

    def _exit(self, reason):
        p = self.pos
        p["reason"] = reason
        self._cancel("stop_id")
        self._cancel("tp1_id")
        q = int(self.portfolio.net_position(p["iid"]))
        if q == 0:
            if "entry_px" not in p:
                self.pos = None
            return
        inst = self.cache.instrument(p["iid"])
        self.submit_order(self.order_factory.market(
            p["iid"], OrderSide.SELL if q > 0 else OrderSide.BUY, inst.make_qty(abs(q)),
            tags=[f"close:{reason}"]))

    def _manage(self, m, h, l, c):
        p, cfg = self.pos, self.c
        if p["iid"] != self.cur_fut.id or "entry_px" not in p \
                or int(self.portfolio.net_position(p["iid"])) == 0:
            return
        side, e0, R = p["side"], p["entry_px"], p["stop_d"]
        if p.get("stop_id") is None and ((side > 0 and l <= p["stop"]) or (side < 0 and h >= p["stop"])):
            return self._exit("stop")
        fav = (h - e0) if side > 0 else (e0 - l)
        p["mfe"] = max(p["mfe"], fav / R)
        if cfg.be_r > 0 and not p["be"] and p["mfe"] >= cfg.be_r:
            be = self._snap(e0)
            if (side > 0 and be < c) or (side < 0 and be > c):
                p["be"] = True
                self._move_stop(be)
        if p["kind"] == "vwap" and m.vwap == m.vwap:
            tgt = m.vwap - side * cfg.target_sd * m.vsd
            if (side > 0 and c >= tgt) or (side < 0 and c <= tgt):
                return self._exit("target")
        if p["kind"] == "pullback" and (c - e0) * side >= cfg.tp2_r * R:
            return self._exit("target")
        if cfg.max_hold_bars > 0 and self.i - p["i0"] >= cfg.max_hold_bars:
            self._exit("time")

    def on_order_filled(self, e: OrderFilled):
        p = self.pos
        if p is None or e.instrument_id != p["iid"]:
            return
        signed = float(e.last_qty) * (1 if e.order_side == OrderSide.BUY else -1)
        p["cash"] -= signed * float(e.last_px)
        p.setdefault("entry_px", float(e.last_px))
        net = int(self.portfolio.net_position(p["iid"]))
        if e.client_order_id == p.get("stop_id") and not p.get("reason"):
            p["reason"] = "be" if p["be"] else "stop"
            self._cancel("tp1_id")
        if e.client_order_id == p.get("tp1_id") and not p.get("reason") and net != 0 \
                and abs(net) < p["n"] and not p["partial"]:
            o = self.cache.order(p["tp1_id"])
            if o is not None and not o.is_open:          # partial target fully filled
                p["partial"] = True
                self._cancel("stop_id")                  # replace with a breakeven stop for the rest
                p["stop"] = self._snap(p["entry_px"])
                p["be"] = True
                self._place_stop(abs(net))
        if not p.get("reason") and "stop_id" not in p and abs(net) == p["n"]:
            self._place_stop(p["n"])
            self._place_tp1()
        if p.get("reason") and net == 0:
            mult = float(self.cache.instrument(p["iid"]).multiplier)
            pnl = p["cash"] * mult
            if pnl < 0 and self.day is not None:
                self.day["losses"] += 1
            self.trades.append(dict(
                kind=p["kind"], side="long" if p["side"] > 0 else "short", opened=p["opened"],
                closed=self._ny(), reason=p["reason"], n=p["n"], partial=p["partial"],
                entry=p["entry_px"], exit=float(e.last_px), stop_init=self._snap(p["entry_px"] - p["side"] * p["stop_d"]),
                risk_usd=p["risk_usd"], pnl=pnl, r_mult=pnl / p["risk_usd"] if p["risk_usd"] else np.nan,
                mfe_r=p["mfe"], scale=p["scale"], adx=p["adx"], slope=p["slope"], vol_pct=p["vol_pct"],
                vwap=p["vwap"], vsd=p["vsd"], atr=p["atr"], bars=self.i - p["i0"], contract=str(p["iid"])))
            self.log.info(f"{p['kind']} {'L' if p['side'] > 0 else 'S'} {p['opened']:%Y-%m-%d %H:%M} -> "
                          f"{self._ny():%H:%M} {p['reason']:7s} n={p['n']} pnl=${pnl:,.0f} "
                          f"({pnl / p['risk_usd']:+.2f}R)")
            self.pos = None

    # ------------------------------------------------------------------ report
    def on_stop(self):
        self._new_day(None)
        t = pd.DataFrame(self.trades)
        d = pd.DataFrame(self.daily)
        eq_end = self._equity()
        eq0 = float(self.start_equity or eq_end)
        stats = {"Total P&L (incl. fees)": f"{eq_end - eq0:,.0f}",
                 "Return": f"{eq_end / eq0 - 1:.2%}",
                 "Max drawdown (intraday)": f"{self.max_dd_usd:,.0f} ({self.max_dd_pct:.2%})"}
        if len(d):
            eq = pd.concat([pd.Series([eq0]), d["equity"].astype(float)], ignore_index=True)
            r = eq.pct_change().dropna()
            ddd = (eq / eq.cummax() - 1).min()
            stats["Max drawdown (daily close)"] = f"{(eq.cummax() - eq).max():,.0f} ({-ddd:.2%})"
            sh = r.mean() / r.std() * math.sqrt(252) if r.std() > 0 else np.nan
            dn = r[r < 0]
            so = r.mean() / math.sqrt((dn ** 2).mean()) * math.sqrt(252) if len(dn) else np.nan
            ann = (eq.iloc[-1] / eq0) ** (252 / max(len(r), 1)) - 1
            stats["Sharpe (daily, ann.)"] = f"{sh:.2f}"
            stats["Sortino (daily, ann.)"] = f"{so:.2f}"
            stats["Return / max DD"] = f"{(eq_end - eq0) / self.max_dd_usd:.2f}" if self.max_dd_usd > 0 else "n/a"
            stats["Calmar (ann. return / daily max DD)"] = f"{ann / -ddd:.2f}" if ddd < 0 else "n/a"
            stats["Days"] = len(d)
            stats["Worst day"] = f"{d['day_pnl'].min():,.0f}"
            stats["Best day"] = f"{d['day_pnl'].max():,.0f}"
            stats["Losing days"] = f"{(d['day_pnl'] < 0).mean():.0%}"
            stats["Days with daily stop hit"] = int(d["stopped"].sum())
            stats["Days halted by DD governor"] = int(d["halted"].sum())
            stats["Days de-risked (scale < 1)"] = int((d["scale"] < 1).sum())
        stats["Trades"] = len(t)
        if len(t):
            gw, gl = t.loc[t.pnl > 0, "pnl"].sum(), -t.loc[t.pnl < 0, "pnl"].sum()
            stats["Win rate"] = f"{(t['pnl'] > 0).mean():.0%}"
            stats["Profit factor"] = f"{gw / gl:.2f}" if gl > 0 else "inf"
            stats["Avg trade"] = f"{t['pnl'].mean():,.0f} ({t['r_mult'].mean():+.2f}R)"
            stats["Worst trade"] = f"{t['pnl'].min():,.0f} ({t['r_mult'].min():+.2f}R)"
            stats["Avg contracts"] = f"{t['n'].mean():.1f}"
            stats["Long / short"] = f"{(t['side'] == 'long').sum()} / {(t['side'] == 'short').sum()}"
            for k, g in t.groupby("kind"):
                stats[f"Kind {k}"] = f"{len(g)} trades, {g['pnl'].sum():,.0f}, win {(g['pnl'] > 0).mean():.0%}"
            for k, g in t.groupby("reason"):
                stats[f"Exit {k}"] = f"{len(g)} trades, {g['pnl'].sum():,.0f}"
        for k, v in stats.items():
            self.log.info(f"{k}: {v}")
        os.makedirs(self.c.output_dir, exist_ok=True)
        t.to_csv(os.path.join(self.c.output_dir, "trades.csv"), index=False)
        d.to_csv(os.path.join(self.c.output_dir, "signals.csv"), index=False)
        pd.Series(stats).to_csv(os.path.join(self.c.output_dir, "summary.csv"), header=False)


# ====================================================================== runner
FIELDS = list(LowDdClFuturesConfig.__struct_fields__)
DEFAULTS = dict(zip(FIELDS, LowDdClFuturesConfig.__struct_defaults__))


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
    p = {**DEFAULTS, **params}
    bars = max(p["vol_lookback"], 3 * p["adx_len"], 3 * p["ema_slow"], p["warmup_bars"])
    trading_days = bars * p["bar_minutes"] / (22 * 60)
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
        strategy_path="lowdd_cl_futures:LowDdClFutures",
        config_path="lowdd_cl_futures:LowDdClFuturesConfig",
        config=dict(params, venue=venue, bar_source=bar_source,
                    trade_start_date=str(pd.Timestamp(start).date()), output_dir=out_dir))
    venue_cfg = BacktestVenueConfig(
        name=venue, oms_type="NETTING", account_type="MARGIN", base_currency="USD",
        starting_balances=[f"{cash:.2f} USD"],
        fee_model=ImportableFeeModelConfig(
            fee_model_path="nautilus_trader.backtest.models:PerContractFeeModel",
            config_path="nautilus_trader.backtest.config:PerContractFeeModelConfig",
            config={"commission": f"{fee:.2f} USD"}))
    # Fills from trade ticks by default (the catalog's bbo-1m quotes go stale inside the minute).
    data = [BacktestDataConfig(catalog_path=catalog_path, data_cls=QuoteTick, instrument_ids=ids,
                               start_time=data_start, end_time=data_end)] if quotes else []
    if bar_source == "EXTERNAL":
        data.append(BacktestDataConfig(catalog_path=catalog_path, data_cls=Bar,
                                       bar_types=[f"{i}-1-MINUTE-LAST-EXTERNAL" for i in ids],
                                       start_time=data_start, end_time=data_end))
    if bar_source != "EXTERNAL" or not quotes:
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
    ap.add_argument("--warmup-days", type=int, default=None)
    ap.add_argument("--cash", type=float, default=100_000.0,
                    help="starting equity; position size scales with it (default 100,000)")
    ap.add_argument("--fee", type=float, default=2.0, help="commission per contract, USD")
    ap.add_argument("--param", action="append", help="key=value (dashes ok), repeatable")
    ap.add_argument("--params-json", default=None)
    ap.add_argument("--venue", default="GLBX")
    ap.add_argument("--bars", default="auto", choices=["auto", "EXTERNAL", "INTERNAL"])
    ap.add_argument("--quotes", action="store_true")
    ap.add_argument("--chunk-size", type=int, default=250_000)
    ap.add_argument("--out", default="results_lowdd")
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args()

    from nautilus_trader.backtest.node import BacktestNode

    params, _ = parse_params(a.param, a.params_json)
    os.makedirs(a.out, exist_ok=True)
    cfg, info = build_config(a.catalog, a.start, a.end, a.cash, a.fee, params, a.out, venue=a.venue,
                             chunk_size=a.chunk_size, log_level=a.log_level, bar_source=a.bars,
                             warmup_days=a.warmup_days, quotes=a.quotes)
    print(f"futures {info['futures']}, bars {info['bar_source']}, data from {info['data_start'][:10]}, "
          f"cash {a.cash:,.0f}")
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
