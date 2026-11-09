"""CL (WTI crude) futures traded off Andrews' pitchfork — standalone NautilusTrader script.

Pivots: a causal ATR zigzag on N-minute bars. A swing high/low is confirmed once price has
reversed pivot-atr x ATR from it (ATR = EWMA true range, half-life atr-halflife bars). So
every pivot is known only after the fact, exactly as it would be live; no look-ahead.

Pitchfork: built from the last three confirmed pivots P0, P1, P2 each time a new pivot is
confirmed. Time is measured in bars, so the fork's lines are straight in bar space.
  median line (ML)  from the handle through the midpoint of P1-P2
  parallels         through P1 and P2, parallel to ML (they sit symmetrically around ML)
  handle by variant:
    andrews   P0
    schiff    (P0 time, midpoint price of P0-P1)
    mschiff   modified Schiff: midpoint of P0-P1 in time and price
A fork is skipped if it is narrower than min-width-atr x ATR, and expires fork-max-bars
bars after P2.

Signals (mode), checked on each bar close inside the trading window:
  revert    Touch-and-hold at an outer parallel, aim for ML. Long when the bar low comes
            within touch-atr x ATR of the lower parallel but closes above it (and below ML);
            short mirror at the upper parallel. Target: the line target-frac of the way from
            the touched parallel to ML (1 = ML, 2 = opposite parallel), re-evaluated every bar
            because the lines slope. Stop: stop-atr x ATR beyond the touched parallel.
  breakout  Fork failure: a fresh close beyond an outer parallel by break-atr x ATR, traded in
            the direction of the break. Exit when price closes back inside the fork
            (reentry), at break-target x fork width from entry (0 = off), or at the stop
            stop-atr x ATR back inside the broken parallel.
  both      either of the above.
slope-filter: any = take everything; with = only trades in the direction the fork slopes
(revert longs off the lower parallel in a rising fork, breakouts in the slope direction).
All modes: max-hold-bars time stop (0 = off), max-trades-day, flat by flat-time.

Session handling follows kalman_cl_futures.py: pivots and forks are built on every bar
around the clock, everything resets when the traded contract rolls, entries 09:30-13:45 ET,
flat 14:20, no entries around the Wednesday 10:30 EIA report or on exchange holidays.

Outputs in --out: trades.csv, signals.csv (daily equity), forks.csv (every fork with its
pivots, for plotting), summary.csv.

  python pitchfork_cl_futures.py --catalog fop_data/catalog --start 2026-01-05 --end 2026-03-31 --out results_pitchfork
  python optimize.py --module pitchfork_cl_futures --catalog fop_data/catalog --grid pitchfork_grid.json ...
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
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

# Shared session helpers live in the Kalman script; reuse them so the two stay in sync.
from kalman_cl_futures import (NS_DAY, NS_MIN, NY, _hm, has_external_bars,  # noqa: E402
                               is_root_future, nyse_holidays)


# ====================================================================== pivots + fork
class Atr:
    """EWMA of true range."""

    def __init__(self, halflife):
        self.alpha = 1.0 - 0.5 ** (1.0 / max(halflife, 1))
        self.reset()

    def reset(self):
        self.value, self.prev_close, self.n = None, None, 0

    def update(self, h, l, c):
        tr = h - l if self.prev_close is None else \
            max(h - l, abs(h - self.prev_close), abs(l - self.prev_close))
        self.value = tr if self.value is None else (1 - self.alpha) * self.value + self.alpha * tr
        self.prev_close = c
        self.n += 1
        return self.value


class ZigZag:
    """Causal swing detector. Pivots are (bar_index, price, kind, ts_ns), kind +1 high / -1 low.

    A candidate extreme becomes a pivot once price reverses by `thr` from it; it is then
    appended with its own bar index (in the past), so the lines are drawn through the true
    swing point even though it is only known now.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.dir = 0                 # 0 undecided, +1 tracking a high, -1 tracking a low
        self.hi = self.lo = None     # (i, price, ts) candidates
        self.pivots = []

    def update(self, i, ts, h, l, thr):
        """Feed one bar; returns the newly confirmed pivot or None."""
        if self.hi is None:
            self.hi, self.lo = (i, h, ts), (i, l, ts)
            return None
        if self.dir == 0:
            if h > self.hi[1]:
                self.hi = (i, h, ts)
            if l < self.lo[1]:
                self.lo = (i, l, ts)
            if self.hi[1] - self.lo[1] < thr:
                return None
            if self.hi[0] > self.lo[0]:          # rallied off the low: low confirmed
                p, self.dir = (self.lo[0], self.lo[1], -1, self.lo[2]), +1
            else:
                p, self.dir = (self.hi[0], self.hi[1], +1, self.hi[2]), -1
            self.pivots.append(p)
            return p
        if self.dir > 0:
            if h >= self.hi[1]:
                self.hi = (i, h, ts)
            elif self.hi[1] - l >= thr:
                p = (self.hi[0], self.hi[1], +1, self.hi[2])
                self.pivots.append(p)
                self.dir, self.lo = -1, (i, l, ts)
                return p
        else:
            if l <= self.lo[1]:
                self.lo = (i, l, ts)
            elif h - self.lo[1] >= thr:
                p = (self.lo[0], self.lo[1], -1, self.lo[2])
                self.pivots.append(p)
                self.dir, self.hi = +1, (i, h, ts)
                return p
        return None


class Fork:
    """Pitchfork lines as functions of bar index."""

    def __init__(self, p0, p1, p2, variant="andrews"):
        (i0, y0), (i1, y1), (i2, y2) = p0[:2], p1[:2], p2[:2]
        if variant == "schiff":
            ai, ay = i0, 0.5 * (y0 + y1)
        elif variant == "mschiff":
            ai, ay = 0.5 * (i0 + i1), 0.5 * (y0 + y1)
        elif variant == "andrews":
            ai, ay = i0, y0
        else:
            raise ValueError(f"variant must be andrews, schiff or mschiff, not {variant!r}")
        mi, my = 0.5 * (i1 + i2), 0.5 * (y1 + y2)
        self.p0, self.p1, self.p2, self.variant = p0, p1, p2, variant
        self.ai, self.ay = ai, ay
        self.slope = (my - ay) / max(mi - ai, 1e-9)   # price per bar
        self.half = abs(y1 - self.ml(i1))             # P1 and P2 sit +-half around ML
        self.width = 2.0 * self.half
        self.i2 = i2

    def ml(self, i):
        return self.ay + self.slope * (i - self.ai)

    def upper(self, i):
        return self.ml(i) + self.half

    def lower(self, i):
        return self.ml(i) - self.half


# ====================================================================== strategy
class PitchforkClFuturesConfig(StrategyConfig, frozen=True):
    venue: str = "GLBX"
    root: str = "CL"
    bar_source: str = "INTERNAL"       # INTERNAL: bars from trades; EXTERNAL: 1-min bars in catalog
    trade_start_date: str = "2026-01-05"
    # pivots / fork
    bar_minutes: int = 15
    atr_halflife: int = 20             # bars
    pivot_atr: float = 3.0             # zigzag reversal in ATRs
    variant: str = "andrews"           # andrews | schiff | mschiff
    min_width_atr: float = 2.0         # skip forks narrower than this
    fork_max_bars: int = 200           # fork expires this many bars after P2 (0 = never)
    warmup_bars: int = 40              # bars after a reset before trading
    # signals
    mode: str = "revert"               # revert | breakout | both
    slope_filter: str = "any"          # any | with
    touch_atr: float = 0.25            # revert: how close the low/high must get to the parallel
    target_frac: float = 1.0           # revert: 1 = median line, 2 = opposite parallel
    break_atr: float = 0.5             # breakout: close beyond the parallel by this many ATRs
    break_target: float = 1.0          # breakout: target in fork widths from entry (0 = none)
    stop_atr: float = 1.0              # stop distance beyond the parallel, in ATRs
    max_hold_bars: int = 0             # time stop (0 = off)
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
    output_dir: str = "results_pitchfork"


class PitchforkClFutures(Strategy):

    def __init__(self, config: PitchforkClFuturesConfig):
        super().__init__(config)
        self.c = c = config
        if c.mode not in ("revert", "breakout", "both"):
            raise ValueError(f"mode must be revert, breakout or both, not {c.mode!r}")
        if c.slope_filter not in ("any", "with"):
            raise ValueError(f"slope_filter must be any or with, not {c.slope_filter!r}")
        Fork((0, 0.0), (1, 1.0), (2, 0.0), c.variant)   # validates the variant name early
        self.venue = Venue(c.venue)
        self.first_day = pd.Timestamp(c.trade_start_date).date()
        self.t_open, self.t_close = _hm(c.session_open), _hm(c.session_close)
        self.t_trade, self.t_last, self.t_flat = _hm(c.trade_start), _hm(c.last_entry), _hm(c.flat_time)
        self.eia = tuple(_hm(x) for x in c.eia_block.split("-")) if c.eia_block else None
        self.atr = Atr(c.atr_halflife)
        self.zz = ZigZag()
        self.fork = None
        self.i = 0                                      # bar index since the last reset
        self.prev_close = None
        self.futs: dict[str, FuturesContract] = {}
        self.cur_fut = None
        self.pos = None
        self.trades, self.daily, self.forks = [], [], []
        self.day = None
        self.holidays: dict[int, set] = {}
        self.last_bar_ns = 0
        self.start_equity = None

    # ------------------------------------------------------------------ setup
    def on_start(self):
        for i in self.cache.instruments(venue=self.venue):
            if type(i) is FuturesContract and is_root_future(i.raw_symbol.value, self.c.root):
                self.futs[i.raw_symbol.value] = i
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
                      f"({self.c.bar_source}), {self.c.variant} fork, mode {self.c.mode}")

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

    def _record_day(self, new_date):
        if self.day is not None and self.day["date"] >= self.first_day:
            f = self.fork
            self.daily.append(dict(date=self.day["date"], equity=self._equity(),
                                   atr=self.atr.value, pivots=len(self.zz.pivots),
                                   fork_slope=f.slope if f else np.nan,
                                   fork_width=f.width if f else np.nan,
                                   contract=self.day["contract"], trades=self.day["trades"]))
        self.day = None if new_date is None else dict(date=new_date, trades=0,
                                                      contract=self.cur_fut.raw_symbol.value
                                                      if self.cur_fut else None)

    def _reset(self):
        self.atr.reset()
        self.zz.reset()
        self.fork, self.i, self.prev_close = None, 0, None

    def _fork_active(self):
        f = self.fork
        return f is not None and (self.c.fork_max_bars <= 0 or self.i - f.i2 <= self.c.fork_max_bars)

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
            self._reset()
            self.log.info(f"pivots reset on {tf.raw_symbol.value}")
        self.last_bar_ns = bar.ts_event
        h, l, c = float(bar.high), float(bar.low), float(bar.close)
        if c <= 0:
            return
        self.i += 1
        i = self.i
        atr = self.atr.update(h, l, c)
        prev_close, self.prev_close = self.prev_close, c
        if self.atr.n >= 5:                              # a few bars before the ATR means anything
            p = self.zz.update(i, bar.ts_event, h, l, self.c.pivot_atr * atr)
            if p is not None and len(self.zz.pivots) >= 3:
                self._new_fork(atr)

        t = self._ny(bar.ts_event)
        if not (self.t_open < t.time() <= self.t_close):
            return                                       # overnight: pivots only
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
                self._manage(h, l, c)
            return
        if t.year not in self.holidays:
            self.holidays[t.year] = nyse_holidays(t.year)
        if t.time() < self.t_trade or t.time() > self.t_last or t.date() in self.holidays[t.year] \
                or self.day["trades"] >= self.c.max_trades_day or i < self.c.warmup_bars \
                or not self._fork_active() \
                or (self.eia is not None and t.weekday() == self.c.eia_weekday
                    and self.eia[0] <= t.time() <= self.eia[1]):
            return
        self._signal(h, l, c, prev_close, atr)

    def _new_fork(self, atr):
        f = Fork(*self.zz.pivots[-3:], variant=self.c.variant)
        ok = f.width >= self.c.min_width_atr * atr
        p0, p1, p2 = f.p0, f.p1, f.p2
        self.forks.append(dict(
            confirmed=self._ny(), contract=self.cur_fut.raw_symbol.value, variant=f.variant,
            used=ok, p0_time=self._ny(p0[3]), p0=p0[1], p1_time=self._ny(p1[3]), p1=p1[1],
            p2_time=self._ny(p2[3]), p2=p2[1], p2_kind="high" if p2[2] > 0 else "low",
            bars_p0_p2=p2[0] - p0[0], slope_per_bar=f.slope, width=f.width, atr=atr))
        if ok:
            self.fork = f

    def _signal(self, h, l, c, prev_close, atr):
        f, i, cfg = self.fork, self.i, self.c
        up, ml, lo = f.upper(i), f.ml(i), f.lower(i)
        slope_sign = 1 if f.slope > 0 else -1 if f.slope < 0 else 0
        allow = (lambda side: True) if cfg.slope_filter == "any" else (lambda side: side == slope_sign)
        if cfg.mode in ("revert", "both"):
            if l <= lo + cfg.touch_atr * atr and lo < c < ml and allow(+1):
                return self._enter(+1, "revert", c, stop=lo - cfg.stop_atr * atr, line="lower")
            if h >= up - cfg.touch_atr * atr and ml < c < up and allow(-1):
                return self._enter(-1, "revert", c, stop=up + cfg.stop_atr * atr, line="upper")
        if cfg.mode in ("breakout", "both") and prev_close is not None:
            b = cfg.break_atr * atr
            pu, pl = f.upper(i - 1), f.lower(i - 1)     # fresh break: last close was not beyond
            tgt = cfg.break_target * f.width
            if c > up + b and prev_close <= pu + b and allow(+1):
                return self._enter(+1, "breakout", c, stop=up - cfg.stop_atr * atr, line="upper",
                                   target=c + tgt if tgt > 0 else None)
            if c < lo - b and prev_close >= pl - b and allow(-1):
                return self._enter(-1, "breakout", c, stop=lo + cfg.stop_atr * atr, line="lower",
                                   target=c - tgt if tgt > 0 else None)

    # ------------------------------------------------------------------ orders
    def _enter(self, side, kind, px, stop, line, target=None):
        n = int(self.c.contracts)
        if n <= 0:
            return
        f = self.fork
        self.pos = dict(side=side, kind=kind, iid=self.cur_fut.id, n=n, stop=stop, target=target,
                        line=line, fork=f, i0=self.i, opened=self._ny(), cash=0.0,
                        ml=f.ml(self.i), upper=f.upper(self.i), lower=f.lower(self.i))
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
            return
        inst = self.cache.instrument(p["iid"])
        self.submit_order(self.order_factory.market(
            p["iid"], OrderSide.SELL if q > 0 else OrderSide.BUY, inst.make_qty(abs(q)),
            tags=[f"close:{reason}"]))

    def _manage(self, h, l, c):
        p = self.pos
        if int(self.portfolio.net_position(p["iid"])) == 0:
            return                                       # entry not filled yet
        side, f, i = p["side"], p["fork"], self.i
        if p.get("stop_id") is None and ((side > 0 and l <= p["stop"]) or (side < 0 and h >= p["stop"])):
            return self._exit("stop")                    # only if the resting stop could not be placed
        if p["kind"] == "revert":
            edge = f.lower(i) if p["line"] == "lower" else f.upper(i)
            goal = edge + self.c.target_frac * (f.ml(i) - edge)
            if (side > 0 and c >= goal) or (side < 0 and c <= goal):
                return self._exit("target")
        else:
            if p["target"] is not None and ((side > 0 and c >= p["target"]) or (side < 0 and c <= p["target"])):
                return self._exit("target")
            if (side > 0 and c < f.upper(i)) or (side < 0 and c > f.lower(i)):
                return self._exit("reentry")             # break failed, back inside the fork
        if self.c.max_hold_bars > 0 and i - p["i0"] >= self.c.max_hold_bars:
            self._exit("time")

    def on_order_filled(self, e: OrderFilled):
        p = self.pos
        if p is None or e.instrument_id != p["iid"]:
            return
        signed = float(e.last_qty) * (1 if e.order_side == OrderSide.BUY else -1)
        p["cash"] -= signed * float(e.last_px)
        p.setdefault("entry_px", float(e.last_px))
        if e.client_order_id == p.get("stop_id") and not p.get("reason"):
            p["reason"] = "stop"
        net = int(self.portfolio.net_position(p["iid"]))
        if not p.get("reason") and "stop_id" not in p and abs(net) == p["n"]:
            self._place_stop()
        if p.get("reason") and int(self.portfolio.net_position(p["iid"])) == 0:
            pnl = p["cash"] * float(self.cache.instrument(p["iid"]).multiplier)
            f = p["fork"]
            self.trades.append(dict(kind=p["kind"], side="long" if p["side"] > 0 else "short",
                                    opened=p["opened"], closed=self._ny(), reason=p["reason"],
                                    n=p["n"], entry=p["entry_px"], exit=float(e.last_px),
                                    line=p["line"], upper=p["upper"], ml=p["ml"], lower=p["lower"],
                                    stop=p["stop"], target=p["target"], variant=f.variant,
                                    fork_p2_time=self._ny(f.p2[3]), fork_width=f.width,
                                    fork_slope=f.slope, pnl=pnl, contract=str(p["iid"])))
            self.log.info(f"{p['kind']} {'L' if p['side'] > 0 else 'S'} {p['opened']:%Y-%m-%d %H:%M} -> "
                          f"{self._ny():%H:%M} {p['reason']:7s} n={p['n']} pnl=${pnl:,.0f}")
            self.pos = None

    def on_stop(self):
        self._record_day(None)
        t = pd.DataFrame(self.trades)
        eq = self._equity()
        stats = {"Total P&L (incl. fees)": f"{eq - float(self.start_equity or eq):,.0f}",
                 "Trades": len(t),
                 "Win rate": f"{(t['pnl'] > 0).mean():.0%}" if len(t) else "n/a",
                 "Avg trade P&L": f"{t['pnl'].mean():,.0f}" if len(t) else "n/a",
                 "Long / short": f"{(t['side'] == 'long').sum()} / {(t['side'] == 'short').sum()}" if len(t) else "0 / 0",
                 "Forks built / used": f"{len(self.forks)} / {sum(f['used'] for f in self.forks)}"}
        if len(t):
            for k, g in t.groupby("kind"):
                stats[f"Kind {k}"] = f"{len(g)} trades, {g['pnl'].sum():,.0f}"
            for r, g in t.groupby("reason"):
                stats[f"Exit {r}"] = f"{len(g)} trades, {g['pnl'].sum():,.0f}"
        for k, v in stats.items():
            self.log.info(f"{k}: {v}")
        os.makedirs(self.c.output_dir, exist_ok=True)
        t.to_csv(os.path.join(self.c.output_dir, "trades.csv"), index=False)
        pd.DataFrame(self.daily).to_csv(os.path.join(self.c.output_dir, "signals.csv"), index=False)
        pd.DataFrame(self.forks).to_csv(os.path.join(self.c.output_dir, "forks.csv"), index=False)
        pd.Series(stats).to_csv(os.path.join(self.c.output_dir, "summary.csv"), header=False)


# ====================================================================== runner
FIELDS = list(PitchforkClFuturesConfig.__struct_fields__)
DEFAULTS = dict(zip(FIELDS, PitchforkClFuturesConfig.__struct_defaults__))


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
        strategy_path="pitchfork_cl_futures:PitchforkClFutures",
        config_path="pitchfork_cl_futures:PitchforkClFuturesConfig",
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
                    help="first trading day; data from --warmup-days earlier builds the first pivots")
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
    ap.add_argument("--out", default="results_pitchfork")
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
