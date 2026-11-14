"""ES (E-mini S&P 500) opening range breakout — standalone NautilusTrader script.

Opening range (OR): high/low of the first or-minutes of the RTH session (09:30 ET), built
from N-minute bars (bar-minutes must divide or-minutes). Nothing is traded before the OR
is complete, so there is no look-ahead.

Range filters (optional): the OR width is compared with ADR, the mean RTH high-low range of
the previous adr-days sessions. Days with OR < min-range-adr x ADR (no real range) or
OR > max-range-adr x ADR (the move already happened) are skipped. 0 switches a filter off.

Entries (entry-type):
  stop   resting STOP_MARKET orders at OR high + buffer and OR low - buffer, placed the moment
         the OR completes; the first to fill cancels the other. Fills on the actual break.
  close  market order after a bar closes beyond OR high + buffer / OR low - buffer.
direction: both | long | short. buffer-ticks is in instrument ticks (ES tick = 0.25).
Entries only until last-entry; unfilled stop entries are cancelled then.

Exits:
  stop    resting reduce-only STOP_MARKET at (breakout level -+ stop-frac x OR width):
          stop-frac 1.0 = opposite side of the range, 0.5 = range midpoint.
  target  resting reduce-only LIMIT at entry +- target-r x initial risk (0 = no target).
  be      optional: once price has gone be-r x risk in favour, the stop moves to entry (0 = off).
  eod     flat at flat-time (default 15:55 ET), also via a timer so it does not wait for bars.
With max-trades-day 2, a stopped-out breakout re-arms the opposite side (each side trades at
most once a day), i.e. a failed-breakout reversal.

Sizing: fixed `contracts`, or risk-usd > 0 = floor(risk-usd / (risk points x multiplier)),
capped at max-contracts.

Roll: trades the nearest future more than roll-days from expiry (ES rolls ~8 days before
the quarterly expiry). Exchange holidays are skipped. Session helpers come from
kalman_cl_futures.py, which must sit beside this file.

Outputs in --out: trades.csv, signals.csv (daily equity + OR stats), summary.csv.

  python orb_es_futures.py --catalog fop_data/catalog --start 2026-01-20 --end 2026-03-31 --out results_orb
  python orb_es_futures.py --catalog fop_data/catalog --param or-minutes=30 --param target-r=2 ...
  python optimize.py --module orb_es_futures --catalog fop_data/catalog --grid orb_grid.json ...
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import deque
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from nautilus_trader.config import StrategyConfig  # noqa: E402
from nautilus_trader.model.data import Bar, BarType  # noqa: E402
from nautilus_trader.model.enums import OrderSide, TimeInForce  # noqa: E402
from nautilus_trader.model.events import OrderFilled, OrderRejected  # noqa: E402
from nautilus_trader.model.identifiers import Venue  # noqa: E402
from nautilus_trader.model.instruments import FuturesContract  # noqa: E402
from nautilus_trader.trading.strategy import Strategy  # noqa: E402

from kalman_cl_futures import (NS_DAY, NS_MIN, NY, _hm, has_external_bars,  # noqa: E402
                               is_root_future, nyse_holidays)


def _add_min(t, minutes):
    return (datetime(2000, 1, 1, t.hour, t.minute) + timedelta(minutes=minutes)).time()


# ====================================================================== strategy
class OrbEsFuturesConfig(StrategyConfig, frozen=True):
    venue: str = "GLBX"
    root: str = "ES"
    bar_source: str = "INTERNAL"       # INTERNAL: bars from trades; EXTERNAL: 1-min bars in catalog
    trade_start_date: str = "2026-01-05"
    bar_minutes: int = 1
    # opening range
    rth_open: str = "09:30"
    rth_close: str = "16:00"
    or_minutes: int = 15
    adr_days: int = 10                 # sessions in the average daily range
    min_range_adr: float = 0.0         # skip if OR < this x ADR (0 = off)
    max_range_adr: float = 0.0         # skip if OR > this x ADR (0 = off)
    # entries
    entry_type: str = "stop"           # stop | close
    direction: str = "both"            # both | long | short
    buffer_ticks: int = 1
    last_entry: str = "12:00"
    max_trades_day: int = 1
    # exits
    stop_frac: float = 1.0             # 1 = opposite side of the OR, 0.5 = OR midpoint
    target_r: float = 0.0              # target in multiples of initial risk (0 = none)
    be_r: float = 0.0                  # move stop to entry after this many R (0 = off)
    flat_time: str = "15:55"
    roll_days: int = 8
    # sizing
    contracts: int = 1
    risk_usd: float = 0.0              # > 0: size by risk instead of `contracts`
    max_contracts: int = 10
    output_dir: str = "results_orb"


class OrbEsFutures(Strategy):

    def __init__(self, config: OrbEsFuturesConfig):
        super().__init__(config)
        self.c = c = config
        if c.entry_type not in ("stop", "close"):
            raise ValueError(f"entry_type must be stop or close, not {c.entry_type!r}")
        if c.direction not in ("both", "long", "short"):
            raise ValueError(f"direction must be both, long or short, not {c.direction!r}")
        if c.or_minutes <= 0 or c.or_minutes % c.bar_minutes:
            raise ValueError("or_minutes must be a positive multiple of bar_minutes")
        self.venue = Venue(c.venue)
        self.first_day = pd.Timestamp(c.trade_start_date).date()
        self.t_open, self.t_close = _hm(c.rth_open), _hm(c.rth_close)
        self.t_or_end = _add_min(self.t_open, c.or_minutes)
        self.t_last, self.t_flat = _hm(c.last_entry), _hm(c.flat_time)
        self.sides = {"both": (1, -1), "long": (1,), "short": (-1,)}[c.direction]
        self.futs: dict[str, FuturesContract] = {}
        self.cur_fut = None
        self.ranges = deque(maxlen=max(c.adr_days, 1))   # completed RTH ranges
        self.day = None
        self.pos = None
        self.entry_ids = {}                               # side -> client_order_id (stop entries)
        self.trades, self.daily = [], []
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
        step = timedelta(minutes=1)
        first = pd.Timestamp(self.clock.utc_now()).floor("1min") + step
        self.clock.set_timer("guard", step, start_time=first.to_pydatetime(), callback=self._on_guard)
        self.log.info(f"{self.c.root}: futures {sorted(self.futs)}, {self.c.bar_minutes}-min bars "
                      f"({self.c.bar_source}), OR {self.c.or_minutes} min, entry {self.c.entry_type}")

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

    def _adr(self):
        return float(np.mean(self.ranges)) if len(self.ranges) >= max(1, self.c.adr_days // 2) else None

    def _close_day(self):
        d = self.day
        if d is None:
            return
        if d["hi"] is not None and d["bars"] >= 30 // self.c.bar_minutes:   # skip near-empty sessions
            self.ranges.append(d["hi"] - d["lo"])
        if d["date"] >= self.first_day:
            self.daily.append(dict(
                date=d["date"], equity=self._equity(), contract=d["contract"],
                or_high=d["or_hi"], or_low=d["or_lo"],
                or_width=(d["or_hi"] - d["or_lo"]) if d["or_hi"] is not None else np.nan,
                adr=d["adr"], status=d["status"], trades=d["trades"],
                rth_high=d["hi"], rth_low=d["lo"], rth_close=d["last"]))
        self.day = None

    def _new_day(self, date):
        self._close_day()
        self.day = dict(date=date, contract=self.cur_fut.raw_symbol.value, bars=0,
                        hi=None, lo=None, last=None, or_hi=None, or_lo=None, or_done=False,
                        adr=self._adr(), status="building", trades=0, sides_done=set())

    def _tradeable_day(self, date):
        if date < self.first_day:
            return False
        if date.year not in self.holidays:
            self.holidays[date.year] = nyse_holidays(date.year)
        return date not in self.holidays[date.year]

    # ------------------------------------------------------------------ guard timer
    def _on_guard(self, _event):
        """Time-based actions that must not wait for a bar: cancel entries, flat by close."""
        now = self._ny()
        if self.entry_ids and now.time() > self.t_last:
            self._cancel_entries()
        if self.pos and not self.pos.get("reason"):
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
            self._cancel_entries()
            self.cur_fut = tf
            self.log.info(f"trading {tf.raw_symbol.value}")
        self.last_bar_ns = bar.ts_event
        t = self._ny(bar.ts_event)
        if not (self.t_open < t.time() <= self.t_close):      # bar closing inside RTH
            return
        if self.day is None or self.day["date"] != t.date():
            self._new_day(t.date())
        d = self.day
        h, l, c = float(bar.high), float(bar.low), float(bar.close)
        d["bars"] += 1
        d["hi"] = h if d["hi"] is None else max(d["hi"], h)
        d["lo"] = l if d["lo"] is None else min(d["lo"], l)
        d["last"] = c
        if self.start_equity is None and t.date() >= self.first_day:
            self.start_equity = self._equity()

        if not d["or_done"]:
            if t.time() <= self.t_or_end:
                d["or_hi"] = h if d["or_hi"] is None else max(d["or_hi"], h)
                d["or_lo"] = l if d["or_lo"] is None else min(d["or_lo"], l)
            if t.time() >= self.t_or_end:
                d["or_done"] = True
                self._on_or_complete(c)
            return

        if t.time() >= self.t_flat:
            if self.pos and not self.pos.get("reason"):
                self._exit("eod")
            return
        if self.pos:
            if not self.pos.get("reason"):
                self._manage(h, l, c)
            return
        if d["status"] != "armed" or t.time() > self.t_last or d["trades"] >= self.c.max_trades_day:
            return
        if self.c.entry_type == "close":
            buf = self._buf()
            for side in self.sides:
                if side in d["sides_done"]:
                    continue
                lvl = d["or_hi"] + buf if side > 0 else d["or_lo"] - buf
                if (side > 0 and c > lvl) or (side < 0 and c < lvl):
                    self._enter_market(side, lvl)
                    break
        elif not self.entry_ids:
            self._arm_stops(c)                              # re-arm after a closed trade

    def _buf(self):
        return self.c.buffer_ticks * float(self.cur_fut.price_increment)

    def _on_or_complete(self, c):
        d = self.day
        if d["or_hi"] is None:
            d["status"] = "no_range"
            return
        if not self._tradeable_day(d["date"]):
            d["status"] = "skip_day"
            return
        width, adr = d["or_hi"] - d["or_lo"], d["adr"]
        if width <= 0:
            d["status"] = "no_range"
            return
        if (self.c.min_range_adr > 0 or self.c.max_range_adr > 0) and adr is None:
            d["status"] = "no_adr"
            return
        if self.c.min_range_adr > 0 and width < self.c.min_range_adr * adr:
            d["status"] = "too_narrow"
            return
        if self.c.max_range_adr > 0 and width > self.c.max_range_adr * adr:
            d["status"] = "too_wide"
            return
        d["status"] = "armed"
        if self.c.entry_type == "stop":
            self._arm_stops(c)

    # ------------------------------------------------------------------ orders
    def _size(self, risk_pts):
        if self.c.risk_usd <= 0:
            return int(self.c.contracts)
        per = risk_pts * float(self.cur_fut.multiplier)
        n = int(self.c.risk_usd // per) if per > 0 else 0
        return max(0, min(n, int(self.c.max_contracts)))

    def _stop_for(self, side):
        d = self.day
        width = d["or_hi"] - d["or_lo"]
        lvl = d["or_hi"] if side > 0 else d["or_lo"]
        inc = float(self.cur_fut.price_increment)               # snap to the tick grid
        return self.cur_fut.make_price(round((lvl - side * self.c.stop_frac * width) / inc) * inc).as_double()

    def _arm_stops(self, last):
        d, buf = self.day, self._buf()
        if self._ny().time() > self.t_last or d["trades"] >= self.c.max_trades_day:
            return
        for side in self.sides:
            if side in d["sides_done"]:
                continue                                    # each side at most once a day
            trig = d["or_hi"] + buf if side > 0 else d["or_lo"] - buf
            if (side > 0 and last >= trig) or (side < 0 and last <= trig):
                continue                                    # already through: no chase
            n = self._size(abs(trig - self._stop_for(side)))
            if n <= 0:
                continue
            o = self.order_factory.stop_market(
                self.cur_fut.id, OrderSide.BUY if side > 0 else OrderSide.SELL,
                self.cur_fut.make_qty(n), trigger_price=self.cur_fut.make_price(trig),
                time_in_force=TimeInForce.DAY, tags=[f"open:{'long' if side > 0 else 'short'}"])
            self.entry_ids[side] = o.client_order_id
            self.submit_order(o)

    def _cancel_entries(self, keep=None):
        for side, oid in list(self.entry_ids.items()):
            if side == keep:
                continue
            o = self.cache.order(oid)
            if o is not None and o.is_open:
                self.cancel_order(o)
            del self.entry_ids[side]

    def _new_pos(self, side, level, n):
        d = self.day
        self.pos = dict(side=side, iid=self.cur_fut.id, n=n, level=level, stop=self._stop_for(side),
                        target=None, opened=self._ny(), cash=0.0, or_hi=d["or_hi"], or_lo=d["or_lo"],
                        be_done=False)
        d["trades"] += 1
        d["sides_done"].add(side)

    def _enter_market(self, side, level):
        n = self._size(abs(level - self._stop_for(side)))
        if n <= 0:
            return
        self._new_pos(side, level, n)
        self.submit_order(self.order_factory.market(
            self.cur_fut.id, OrderSide.BUY if side > 0 else OrderSide.SELL,
            self.cur_fut.make_qty(n), tags=[f"open:{'long' if side > 0 else 'short'}"]))

    def _place_brackets(self):
        p = self.pos
        inst = self.cache.instrument(p["iid"])
        close_side = OrderSide.SELL if p["side"] > 0 else OrderSide.BUY
        p["risk"] = abs(p["entry_px"] - p["stop"])
        so = self.order_factory.stop_market(
            p["iid"], close_side, inst.make_qty(p["n"]), trigger_price=inst.make_price(p["stop"]),
            reduce_only=True, tags=["close:stop"])
        p["stop_id"] = so.client_order_id
        self.submit_order(so)
        if self.c.target_r > 0 and p["risk"] > 0:
            inc = float(inst.price_increment)
            raw = p["entry_px"] + p["side"] * self.c.target_r * p["risk"]
            p["target"] = inst.make_price(round(raw / inc) * inc).as_double()
            to = self.order_factory.limit(
                p["iid"], close_side, inst.make_qty(p["n"]), price=inst.make_price(p["target"]),
                reduce_only=True, tags=["close:target"])
            p["target_id"] = to.client_order_id
            self.submit_order(to)

    def _cancel_brackets(self, keep=None):
        for k in ("stop_id", "target_id"):
            oid = self.pos.get(k)
            if oid is None or oid == keep:
                continue
            o = self.cache.order(oid)
            if o is not None and o.is_open:
                self.cancel_order(o)

    def _move_stop(self, px):
        p = self.pos
        o = self.cache.order(p["stop_id"]) if p.get("stop_id") else None
        if o is None or not o.is_open:
            return
        inst = self.cache.instrument(p["iid"])
        p["stop"] = px
        self.modify_order(o, trigger_price=inst.make_price(px))

    def on_order_rejected(self, e: OrderRejected):
        for side, oid in list(self.entry_ids.items()):
            if e.client_order_id == oid:                    # entry trigger already in the market
                del self.entry_ids[side]
                return
        p = self.pos
        if p and e.client_order_id == p.get("stop_id"):    # stop already through the market
            p["stop_id"] = None
            if not p.get("reason"):
                self._exit("stop")

    def _exit(self, reason):
        p = self.pos
        p["reason"] = reason
        self._cancel_brackets()
        q = int(self.portfolio.net_position(p["iid"]))
        if q == 0:
            if "entry_px" not in p:                          # never filled
                self.pos = None
            return
        inst = self.cache.instrument(p["iid"])
        self.submit_order(self.order_factory.market(
            p["iid"], OrderSide.SELL if q > 0 else OrderSide.BUY, inst.make_qty(abs(q)),
            tags=[f"close:{reason}"]))

    def _manage(self, h, l, c):
        p = self.pos
        if "entry_px" not in p or int(self.portfolio.net_position(p["iid"])) == 0:
            return
        side = p["side"]
        if p.get("stop_id") is None and ((side > 0 and l <= p["stop"]) or (side < 0 and h >= p["stop"])):
            return self._exit("stop")                      # only if the resting stop is missing
        if self.c.be_r > 0 and not p["be_done"] and p["risk"] > 0:
            fav = (h - p["entry_px"]) if side > 0 else (p["entry_px"] - l)
            if fav >= self.c.be_r * p["risk"]:
                p["be_done"] = True
                self._move_stop(p["entry_px"])

    def on_order_filled(self, e: OrderFilled):
        side_hit = next((s for s, oid in self.entry_ids.items() if oid == e.client_order_id), None)
        if side_hit is not None and self.pos is None:      # a stop entry triggered
            self._cancel_entries(keep=side_hit)
            del self.entry_ids[side_hit]
            lvl = self.day["or_hi"] + self._buf() if side_hit > 0 else self.day["or_lo"] - self._buf()
            o = self.cache.order(e.client_order_id)
            self._new_pos(side_hit, lvl, int(o.quantity))
        p = self.pos
        if p is None or e.instrument_id != p["iid"]:
            return
        signed = float(e.last_qty) * (1 if e.order_side == OrderSide.BUY else -1)
        p["cash"] -= signed * float(e.last_px)
        p.setdefault("entry_px", float(e.last_px))
        if not p.get("reason"):
            if e.client_order_id == p.get("stop_id"):
                p["reason"] = "be" if p["be_done"] else "stop"
            elif e.client_order_id == p.get("target_id"):
                p["reason"] = "target"
            if p.get("reason"):
                self._cancel_brackets(keep=e.client_order_id)
        net = int(self.portfolio.net_position(p["iid"]))
        if not p.get("reason") and "stop_id" not in p and abs(net) == p["n"]:
            self._place_brackets()
        if p.get("reason") and net == 0:
            mult = float(self.cache.instrument(p["iid"]).multiplier)
            pnl = p["cash"] * mult
            risk = p.get("risk") or np.nan
            self.trades.append(dict(
                date=p["opened"].date(), side="long" if p["side"] > 0 else "short",
                opened=p["opened"], closed=self._ny(), reason=p["reason"], n=p["n"],
                level=p["level"], entry=p["entry_px"], exit=float(e.last_px),
                or_high=p["or_hi"], or_low=p["or_lo"], or_width=p["or_hi"] - p["or_lo"],
                stop=p["stop"], target=p["target"], risk_pts=risk,
                r_multiple=(p["cash"] / p["n"]) / risk if risk and risk > 0 else np.nan,
                pnl=pnl, contract=str(p["iid"])))
            self.log.info(f"{'L' if p['side'] > 0 else 'S'} {p['opened']:%Y-%m-%d %H:%M} -> "
                          f"{self._ny():%H:%M} {p['reason']:6s} n={p['n']} pnl=${pnl:,.0f}")
            self.pos = None

    def on_stop(self):
        self._close_day()
        t = pd.DataFrame(self.trades)
        days = pd.DataFrame(self.daily)
        eq = self._equity()
        stats = {"Total P&L (incl. fees)": f"{eq - float(self.start_equity or eq):,.0f}",
                 "Trades": len(t),
                 "Win rate": f"{(t['pnl'] > 0).mean():.0%}" if len(t) else "n/a",
                 "Avg trade P&L": f"{t['pnl'].mean():,.0f}" if len(t) else "n/a",
                 "Avg R": f"{t['r_multiple'].mean():.2f}" if len(t) else "n/a",
                 "Long / short": f"{(t['side'] == 'long').sum()} / {(t['side'] == 'short').sum()}" if len(t) else "0 / 0"}
        if len(days):
            for s, g in days.groupby("status"):
                stats[f"Days {s}"] = len(g)
        if len(t):
            for s, g in t.groupby("side"):
                stats[f"Side {s}"] = f"{len(g)} trades, {g['pnl'].sum():,.0f}"
            for r, g in t.groupby("reason"):
                stats[f"Exit {r}"] = f"{len(g)} trades, {g['pnl'].sum():,.0f}"
        for k, v in stats.items():
            self.log.info(f"{k}: {v}")
        os.makedirs(self.c.output_dir, exist_ok=True)
        t.to_csv(os.path.join(self.c.output_dir, "trades.csv"), index=False)
        days.to_csv(os.path.join(self.c.output_dir, "signals.csv"), index=False)
        pd.Series(stats).to_csv(os.path.join(self.c.output_dir, "summary.csv"), header=False)


# ====================================================================== runner
FIELDS = list(OrbEsFuturesConfig.__struct_fields__)
DEFAULTS = dict(zip(FIELDS, OrbEsFuturesConfig.__struct_defaults__))


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
    if warmup_days is None:   # calendar days to fill the ADR window before the first trade day
        warmup_days = int(math.ceil(params.get("adr_days", DEFAULTS["adr_days"]) * 1.5)) + 3
    data_start = (pd.Timestamp(start) - pd.Timedelta(days=warmup_days)).isoformat()
    data_end = (pd.Timestamp(end) + pd.Timedelta(days=1)).isoformat()
    ids = [str(f.id) for f in futs]
    strategy = ImportableStrategyConfig(
        strategy_path="orb_es_futures:OrbEsFutures",
        config_path="orb_es_futures:OrbEsFuturesConfig",
        config=dict(params, venue=venue, bar_source=bar_source,
                    trade_start_date=str(pd.Timestamp(start).date()), output_dir=out_dir))
    venue_cfg = BacktestVenueConfig(
        name=venue, oms_type="NETTING", account_type="MARGIN", base_currency="USD",
        starting_balances=[f"{cash:.2f} USD"],
        fee_model=ImportableFeeModelConfig(
            fee_model_path="nautilus_trader.backtest.models:PerContractFeeModel",
            config_path="nautilus_trader.backtest.config:PerContractFeeModelConfig",
            config={"commission": f"{fee:.2f} USD"}))
    # Fills come from trade ticks by default: the catalog's quotes are bbo-1m samples, which go
    # stale inside the minute and can trigger stops on a quote the market had already left.
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
    ap.add_argument("--start", default="2026-01-20",
                    help="first trading day; earlier data (--warmup-days) only fills the ADR window")
    ap.add_argument("--end", default="2026-03-31")
    ap.add_argument("--warmup-days", type=int, default=None,
                    help="calendar days of data before --start (default: from adr-days)")
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
    ap.add_argument("--out", default="results_orb")
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
