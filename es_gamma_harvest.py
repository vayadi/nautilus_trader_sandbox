"""ES gamma harvesting driven by dealer gamma exposure (GEX) — standalone NautilusTrader script.

Idea
  Owning an option straddle and delta-hedging it with futures "harvests" gamma: every hedge
  sells after a rise and buys after a fall, so the hedges earn roughly
      0.5 x gamma x (realized move)^2
  while the options lose theta, which is priced off implied vol. The trade wins when realized
  vol during the hold beats the implied vol paid, and the short version wins when it doesn't.

  Dealer GEX says which way that is likely to go:
    dealers SHORT gamma (spot below the zero-gamma flip): their hedging chases the move, so
        realized vol tends to expand -> BUY the ATM straddle and scalp it (long gamma).
    dealers LONG gamma (spot above the flip): their hedging fades moves and pins price near
        big strikes -> optionally SELL the straddle, or a strangle at the put/call walls,
        and delta-hedge it (short gamma, theta harvest).
  An implied-vs-realized filter keeps it from paying up for vol (iv-rv-max) or selling it
  cheap (iv-rv-min).

GEX (same math as es_gex_futures.py, imported from it)
  Every `rebalance-minutes` in RTH the ES options chain (quarterly ES + weekly EW/EW1-4) is
  rebuilt from cached quotes, a Black-76 smile is fitted per expiry, and with open interest
  (or the traded-volume proxy) it computes net dealer GEX, the flip level, the signed cushion
  to the flip and the call / put walls. regime = long_gamma / short_gamma / neutral with
  gex-buffer as the dead band.

Position cycle
  entry   at a rebalance between entry-start and entry-end, flat, regime + vol filter agree.
          Expiry with min-dte <= DTE <= max-dte nearest target-dte; strike nearest the option's
          own underlying future; both legs quoted with spread <= entry-max-spread of mid.
          Size: `straddles` (calls = puts = straddles).
  hedge   on every 1-min bar of the option's underlying future between hedge-start and
          hedge-end: net delta (Black-76, legs' current implied vols) in ES contracts; if
          |delta| >= hedge-band, trade futures back to round(delta) = 0. Unhedged overnight
          unless hedge-overnight=1 (CME futures trade ~23h).
  exit    take-profit / stop-loss on cycle P&L (options + hedges) as a share of the premium,
          regime turning against the position (exit-on-regime-change), max-hold-days, and
          always before expiry: from close-time on the session with DTE <= exit-dte.

Outputs in --out: trades.csv (one row per straddle cycle: option P&L, hedge P&L, fees, IV
paid, realized vol during the hold), signals.csv (daily equity, regime, GEX, IV, RV, delta),
summary.csv, fills.csv, account.csv.

Run (volume proxy for open interest, like es_gex_futures.py)
  python es_gamma_harvest.py --catalog fop_data/catalog --start 2026-01-12 --end 2026-03-13 \
      --oi-source volume --out results_gamma
  python es_gamma_harvest.py ... --param side=both --param short-strikes=walls
Optimize
  python optimize.py --module es_gamma_harvest --catalog fop_data/catalog --grid gamma_harvest_grid.json \
      --start 2026-01-12 --end 2026-02-27 --fixed oi-source=volume --workers 4 --target sharpe --out opt_gamma

es_gex_futures.py must sit beside this file.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from bisect import bisect_right
from collections import deque
from datetime import datetime, time, timedelta

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from nautilus_trader.config import StrategyConfig  # noqa: E402
from nautilus_trader.model.data import Bar, BarType  # noqa: E402
from nautilus_trader.model.enums import OptionKind, OrderSide  # noqa: E402
from nautilus_trader.model.events import OrderDenied, OrderFilled, OrderRejected  # noqa: E402
from nautilus_trader.model.identifiers import Venue  # noqa: E402
from nautilus_trader.model.instruments import FuturesContract, OptionContract  # noqa: E402
from nautilus_trader.trading.strategy import Strategy  # noqa: E402

from es_gex_futures import (LONG_G, NEUTRAL, NY, SHORT_G, _d1, _hm, build_volume_oi,  # noqa: E402
                            classify, gamma_cushion, gamma_walls, gex_curve, has_external_bars,
                            implied_vol, is_root_future, norm_cdf, norm_pdf, nyse_holidays,
                            select_instruments, smile_for_expiry)

NS_MIN = 60_000_000_000
NS_DAY = 86_400_000_000_000
BARS_PER_RTH = 78                      # 5-minute returns in a 09:30-16:00 session


def b76_delta(F, K, T, vol, is_call, df):
    """Black-76 delta per unit of underlying (premium paid up front)."""
    d1, _ = _d1(F, K, T, vol)
    return df * norm_cdf(d1) if is_call else -df * norm_cdf(-d1)


def b76_gamma(F, K, T, vol, df):
    d1, sT = _d1(F, K, T, vol)
    return df * norm_pdf(d1) / (F * sT)


def ann_vol(sum_r2, n, rth_var_share):
    """Annualized vol from n RTH 5-min log returns, scaled up to the whole 24h day."""
    if n < 10:
        return float("nan")
    return math.sqrt(sum_r2 / n * BARS_PER_RTH / rth_var_share * 252.0)


# ====================================================================== strategy
class EsGammaHarvestConfig(StrategyConfig, frozen=True):
    venue: str = "GLBX"
    root: str = "ES"
    bar_source: str = "INTERNAL"
    trade_start_date: str = "2026-01-12"
    # GEX regime (see es_gex_futures.py)
    rebalance_minutes: int = 15
    gex_buffer: float = 0.0025
    gex_max_dte: int = 60
    gex_range: float = 0.08
    gex_steps: int = 81
    strike_pct: float = 0.13
    max_rel_spread: float = 0.35
    stale_minutes: int = 30
    rate: float = 0.04
    oi_path: str = ""
    oi_source: str = "file"            # file | unit (the runner turns `volume` into a file)
    # what to trade
    side: str = "long"                 # long: buy straddles when dealers are short gamma
                                       # short: sell when dealers are long gamma; both
    short_strikes: str = "atm"         # atm straddle | walls: strangle at the put / call walls
    wall_max_dist: float = 0.03        # walls further than this from spot fall back to ATM
    iv_rv_max: float = 1.5             # buy only if straddle IV <= iv_rv_max x RV (0 = off)
    iv_rv_min: float = 1.2             # sell only if straddle IV >= iv_rv_min x RV (0 = off)
    rv_days: int = 5                   # sessions of 5-min returns in RV
    rth_var_share: float = 0.75        # share of daily variance in RTH (as es_gex_futures.py)
    min_dte: int = 2
    max_dte: int = 10
    target_dte: int = 5
    entry_max_spread: float = 0.10     # leg bid/ask spread <= this share of mid
    straddles: int = 10
    # timing (New York)
    entry_start: str = "10:00"
    entry_end: str = "15:00"
    max_entries_day: int = 1
    hedge_start: str = "09:31"
    hedge_end: str = "15:59"
    hedge_overnight: int = 0
    close_time: str = "15:45"
    # hedging
    hedge_band: float = 1.0            # |net delta| in ES contracts that triggers a hedge
    # exits
    exit_dte: int = 1                  # close on the session where DTE <= exit_dte (>= 0)
    take_profit: float = 0.0           # cycle P&L >= this x premium (0 = off)
    stop_loss: float = 0.5             # cycle P&L <= -this x premium (0 = off)
    max_hold_days: int = 0             # 0 = hold until exit_dte
    exit_on_regime_change: int = 1     # exit when the regime turns against the position
    roll_days: int = 8
    output_dir: str = "results_gamma"


class EsGammaHarvest(Strategy):

    def __init__(self, config: EsGammaHarvestConfig):
        super().__init__(config)
        self.c = c = config
        if c.side not in ("long", "short", "both"):
            raise ValueError(f"side must be long, short or both, not {c.side!r}")
        if c.short_strikes not in ("atm", "walls"):
            raise ValueError(f"short_strikes must be atm or walls, not {c.short_strikes!r}")
        if c.min_dte <= c.exit_dte:
            raise ValueError("min_dte must be larger than exit_dte, or every entry exits at once")
        self.venue = Venue(c.venue)
        self.first_day = pd.Timestamp(c.trade_start_date).date()
        self.t_e0, self.t_e1 = _hm(c.entry_start), _hm(c.entry_end)
        self.t_h0, self.t_h1 = _hm(c.hedge_start), _hm(c.hedge_end)
        self.t_close = _hm(c.close_time)
        self.futs: dict[str, FuturesContract] = {}
        self.chain: list[tuple] = []        # (OptionContract, underlying future symbol, K, is_call, expiry date)
        self.oi: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.sig = None
        self.book = None                    # the open straddle cycle
        self.trades, self.daily = [], []
        self.holidays: dict[int, set] = {}
        self.regime_count = {LONG_G: 0, SHORT_G: 0, NEUTRAL: 0}
        self.gex_runs = self.gex_ok = 0
        self.start_equity = None
        self.realized = 0.0
        self.last_bar_ns = 0
        # realized vol from 5-min RTH returns of the traded future
        self.rv_days = deque(maxlen=max(int(c.rv_days), 1))
        self.rv_day = None                  # [date, sum r2, n]
        self.rv_last = None                 # (instrument id, last 5-min close)
        self.day = None                     # date, entries, hedges

    # ------------------------------------------------------------------ setup
    def on_start(self):
        insts = self.cache.instruments(venue=self.venue)
        for i in insts:
            if type(i) is FuturesContract and is_root_future(i.raw_symbol.value, self.c.root):
                self.futs[i.raw_symbol.value] = i
        by_exp = sorted(self.futs.values(), key=lambda f: f.expiration_ns)
        for i in insts:
            if type(i) is not OptionContract:
                continue
            und = i.underlying
            if und not in self.futs:
                if not und.startswith(self.c.root) or is_root_future(und, self.c.root):
                    continue
                cand = [f for f in by_exp if f.expiration_ns >= i.expiration_ns - NS_DAY]
                if not cand:
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
        if self.c.oi_source != "unit" and not self.oi:
            self.log.warning("No open interest: no GEX regime, so the strategy will not trade")
        step = timedelta(minutes=int(self.c.rebalance_minutes))
        first = pd.Timestamp(self.clock.utc_now()).floor(f"{int(self.c.rebalance_minutes)}min") + step
        self.clock.set_timer("gex", step, start_time=first.to_pydatetime(), callback=self._on_timer)
        exps = sorted({r[4] for r in self.chain})
        self.log.info(f"{self.c.root}: futures {sorted(self.futs)}, {len(self.chain)} options on "
                      f"{len(exps)} expiries ({exps[0]} .. {exps[-1]}), OI for {len(self.oi)} symbols")

    # ------------------------------------------------------------------ helpers
    def _ny(self, ns=None):
        ns = self.clock.timestamp_ns() if ns is None else ns
        return pd.Timestamp(ns, tz="UTC").tz_convert(NY).to_pydatetime().replace(tzinfo=None)

    def _bar_type(self, iid):
        return BarType.from_str(f"{iid}-1-MINUTE-LAST-{self.c.bar_source}")

    def _trade_future(self):
        now = self.clock.timestamp_ns()
        live = [f for f in self.futs.values() if f.expiration_ns > now + self.c.roll_days * NS_DAY]
        if not live:
            live = [f for f in self.futs.values() if f.expiration_ns > now + NS_DAY]
        return min(live, key=lambda f: f.expiration_ns) if live else None

    def _price(self, iid):
        q = self.cache.quote_tick(iid)
        if q is not None and q.bid_price > 0 and q.ask_price > q.bid_price:
            return 0.5 * (float(q.bid_price) + float(q.ask_price))
        b = self.cache.bar(self._bar_type(iid))
        return float(b.close) if b is not None else float("nan")

    def _quote(self, iid, max_age_ns=None):
        """(bid, ask) of a fresh two-sided quote, else None."""
        q = self.cache.quote_tick(iid)
        if q is None:
            return None
        if max_age_ns is not None and self.clock.timestamp_ns() - q.ts_event > max_age_ns:
            return None
        bid, ask = float(q.bid_price), float(q.ask_price)
        return (bid, ask) if bid > 0 and ask > bid else None

    def _oi(self, sym, now_ns):
        if self.c.oi_source == "unit":
            return 1.0
        rec = self.oi.get(sym)
        if rec is None:
            return 0.0
        i = bisect_right(rec[0], now_ns) - 1
        return float(rec[1][i]) if i >= 0 else 0.0

    def _cash(self):
        acc = self.portfolio.account(self.venue)
        return float(acc.balance_total(acc.base_currency).as_double()) if acc is not None else float("nan")

    def _equity(self):
        """Starting cash + closed cycles + the open cycle marked at mid. Tracked here rather than
        read from the Nautilus portfolio, whose margin-account marks of options were off."""
        if self.start_equity is None:
            return self._cash()
        eq = self.start_equity + self.realized
        b = self.book
        if b is not None:
            m = self._cycle_pnl(self._price(b["hedge_id"]))
            if np.isfinite(m):
                b["last_mark"] = m
            eq += b.get("last_mark", 0.0)
        return eq

    def _is_holiday(self, d):
        if d.year not in self.holidays:
            self.holidays[d.year] = nyse_holidays(d.year)
        return d in self.holidays[d.year]

    @staticmethod
    def _T(exp, now):
        end = datetime(exp.year, exp.month, exp.day, 16, 0)
        return max((end - now).total_seconds(), 60.0) / (365.0 * 86400.0)

    def _rv(self):
        if len(self.rv_days) < max(2, (self.rv_days.maxlen + 1) // 2):
            return float("nan")
        s = sum(d[1] for d in self.rv_days)
        n = sum(d[2] for d in self.rv_days)
        return ann_vol(s, n, self.c.rth_var_share)

    # ------------------------------------------------------------------ GEX (as es_gex_futures)
    def compute_gex(self, now):
        now_ns = self.clock.timestamp_ns()
        today = now.date()
        groups, prices = {}, {}
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
        items, front = [], (10_000, float("nan"))
        stale = self.c.stale_minutes * NS_MIN
        for (und, exp), rows in groups.items():
            F = prices[und]
            bid, ask = [], []
            for inst, _, _ in rows:
                q = self._quote(inst.id, stale)
                bid.append(q[0] if q else 0.0)
                ask.append(q[1] if q else 0.0)
            T = self._T(exp, now)
            df = math.exp(-self.c.rate * T)
            strikes = np.array([r[1] for r in rows])
            is_call = np.array([r[2] for r in rows])
            sm = smile_for_expiry(F, T, df, strikes, is_call, bid, ask, self.c.max_rel_spread)
            if sm is None:
                continue
            dte = (exp - today).days
            if dte >= 1 and dte < front[0]:
                front = (dte, sm["atm_iv"])
            oi = np.array([self._oi(r[0].raw_symbol.value, now_ns) for r in rows])
            items.append(dict(F=F, T=T, df=df, strikes=strikes, is_call=is_call, iv=sm["smile_iv"], oi=oi))
        if not items:
            return None
        self.gex_runs += 1
        nan = float("nan")
        if not any((it["oi"] > 0).any() for it in items):
            return dict(regime=NEUTRAL, cushion=nan, flip=nan, call_wall=nan, put_wall=nan,
                        gex=nan, atm_iv=front[1], ts=now_ns)
        self.gex_ok += 1
        xs = np.linspace(-self.c.gex_range, self.c.gex_range, int(self.c.gex_steps))
        curve = gex_curve(items, xs)
        cushion = gamma_cushion(xs, curve)
        tf = self._trade_future()
        spot = self._price(tf.id) if tf is not None else nan
        cw, pw = gamma_walls(items)
        return dict(regime=classify(cushion, self.c.gex_buffer), cushion=cushion,
                    flip=spot * (1.0 - cushion) if np.isfinite(cushion) else nan,
                    call_wall=cw, put_wall=pw, gex=float(curve[int(np.argmin(np.abs(xs)))]),
                    atm_iv=front[1], ts=now_ns)

    # ------------------------------------------------------------------ timer: regime, entries, exits
    def _on_timer(self, _event):
        now = self._ny()
        if self.book and not self.book["exiting"]:
            self._time_exits(now)
        if not (time(9, 30) <= now.time() <= time(16, 0)):
            return
        if self.clock.timestamp_ns() - self.last_bar_ns > 5 * NS_MIN:
            return                                       # no market (holiday / weekend / halt)
        sig = self.compute_gex(now)
        if sig is None:
            return
        self.sig = sig
        if now.date() < self.first_day:
            return
        self.regime_count[sig["regime"]] += 1
        if self.book:
            if not self.book["exiting"] and self.c.exit_on_regime_change:
                bad = SHORT_G if self.book["dir"] < 0 else LONG_G   # short options need dealers long gamma
                if sig["regime"] == bad:
                    self._exit("regime")
            return
        if self.day is None or self.day["entries"] >= self.c.max_entries_day:
            return
        if not (self.t_e0 <= now.time() <= self.t_e1) or self._is_holiday(now.date()):
            return
        self._maybe_enter(now, sig)

    def _time_exits(self, now):
        b = self.book
        today = now.date()
        if now.time() >= self.t_close and not self._is_holiday(today) and \
                (b["exp"] - today).days <= self.c.exit_dte:
            return self._exit("expiry")
        if (b["exp"] - today).days < 0 or (b["exp"] == today and now.time() >= time(15, 55)):
            return self._exit("expiry")
        if self.c.max_hold_days > 0 and now.time() >= self.t_close and \
                (today - b["opened"].date()).days >= self.c.max_hold_days:
            return self._exit("max_hold")

    def _maybe_enter(self, now, sig):
        want = 0
        if sig["regime"] == SHORT_G and self.c.side in ("long", "both"):
            want = +1                                   # dealers short gamma -> own gamma
        elif sig["regime"] == LONG_G and self.c.side in ("short", "both"):
            want = -1                                   # dealers long gamma -> sell gamma
        if want == 0:
            return
        pick = self._pick_options(now, sig, want)
        if pick is None:
            return
        rv, iv = self._rv(), pick[6]                     # IV of the straddle actually traded
        if want > 0 and self.c.iv_rv_max > 0 and not (np.isfinite(rv) and iv <= self.c.iv_rv_max * rv):
            return
        if want < 0 and self.c.iv_rv_min > 0 and not (np.isfinite(rv) and iv >= self.c.iv_rv_min * rv):
            return
        n = int(self.c.straddles)
        if n <= 0:
            return
        put, call, und, exp, F, prem, ivs = pick
        hedge = self.futs[und]
        self.book = dict(dir=want, target={put.id: want * n, call.id: want * n},
                         n=n, hedge_id=hedge.id, exp=exp, opened=now, opened_ns=self.clock.timestamp_ns(),
                         put=put, call=call, K_put=float(put.strike_price), K_call=float(call.strike_price),
                         F_entry=F, premium=prem, iv_entry=ivs, atm_iv=sig["atm_iv"], rv_entry=rv,
                         regime=sig["regime"], cushion=sig["cushion"], flip=sig["flip"],
                         call_wall=sig["call_wall"], put_wall=sig["put_wall"],
                         opt_cash=0.0, hedge_cash=0.0, fees=0.0, hedges=0, hedge_qty=0,
                         rv_r2=0.0, rv_n=0, exiting=False, reason=None, pending=0,
                         opt_entry_px={}, opt_exit_px={})
        self.day["entries"] += 1
        side = OrderSide.BUY if want > 0 else OrderSide.SELL
        for inst in (put, call):
            self._submit(inst.id, side, n, f"open:{'long' if want > 0 else 'short'}")
        kind = "buy" if want > 0 else "sell"
        self.log.info(f"{kind} {n}x {put.raw_symbol.value}/{call.raw_symbol.value} exp {exp} F={F:.2f} "
                      f"prem={prem:.2f} iv={iv:.1%} rv={rv:.1%} cushion={sig['cushion']:+.2%}")

    def _pick_options(self, now, sig, want):
        """(put, call, und, exp, F, premium pts per straddle, mid IV) or None."""
        today = now.date()
        stale = self.c.stale_minutes * NS_MIN
        by_key = {}
        for inst, und, k, is_call, exp in self.chain:
            dte = (exp - today).days
            if not (self.c.min_dte <= dte <= self.c.max_dte) or inst.expiration_ns <= self.clock.timestamp_ns():
                continue
            f = self.futs.get(und)
            if f is None or f.expiration_ns < inst.expiration_ns - NS_DAY:
                continue
            by_key.setdefault((und, exp, inst.raw_symbol.value.split(" ")[0]), {}) \
                .setdefault(k, {})["C" if is_call else "P"] = inst
        best = None
        for (und, exp, fam), strikes in by_key.items():
            F = self._price(self.futs[und].id)
            if not np.isfinite(F):
                continue
            ks = sorted(strikes)
            atm = min(ks, key=lambda k: abs(k - F))
            kp = kc = atm
            if want < 0 and self.c.short_strikes == "walls":
                pw, cw = sig["put_wall"], sig["call_wall"]
                if np.isfinite(pw) and pw < F and (1 - pw / F) <= self.c.wall_max_dist:
                    kp = min((k for k in ks if "P" in strikes[k]), key=lambda k: abs(k - pw), default=atm)
                if np.isfinite(cw) and cw > F and (cw / F - 1) <= self.c.wall_max_dist:
                    kc = min((k for k in ks if "C" in strikes[k]), key=lambda k: abs(k - cw), default=atm)
            put, call = strikes[kp].get("P"), strikes[kc].get("C")
            if put is None or call is None:
                continue
            qp, qc = self._quote(put.id, stale), self._quote(call.id, stale)
            if qp is None or qc is None:
                continue
            mp_, mc = 0.5 * sum(qp), 0.5 * sum(qc)
            if (qp[1] - qp[0]) > self.c.entry_max_spread * mp_ or (qc[1] - qc[0]) > self.c.entry_max_spread * mc:
                continue
            T = self._T(exp, now)
            df = math.exp(-self.c.rate * T)
            ivp = float(implied_vol(mp_, F, kp, T, False, df))
            ivc = float(implied_vol(mc, F, kc, T, True, df))
            if not (np.isfinite(ivp) and np.isfinite(ivc)):
                continue
            dte = (exp - today).days
            spread = (qp[1] - qp[0] + qc[1] - qc[0]) / (mp_ + mc)
            score = (abs(dte - self.c.target_dte), spread)
            if best is None or score < best[0]:
                best = (score, (put, call, und, exp, F, mp_ + mc, 0.5 * (ivp + ivc)))
        return best[1] if best else None

    # ------------------------------------------------------------------ bars: RV, hedging, P&L exits
    def on_bar(self, bar: Bar):
        if float(bar.volume) <= 0:
            return
        iid = bar.bar_type.instrument_id
        t = self._ny(bar.ts_event)
        tf = self._trade_future()
        if tf is not None and iid == tf.id:
            self.last_bar_ns = bar.ts_event
            self._update_rv(iid, t, float(bar.close))
            if time(9, 31) <= t.time() <= time(16, 0) and (self.day is None or self.day["date"] != t.date()):
                self._new_day(t.date())
        b = self.book
        if b is None or iid != b["hedge_id"]:
            return
        if b["exiting"]:
            return
        if b["pending"] or any(self._pos(i) != q for i, q in b["target"].items()):
            return                                       # entry not filled yet
        in_rth = self.t_h0 <= t.time() <= self.t_h1
        if not in_rth and not self.c.hedge_overnight:
            return
        if t.date() < self.first_day:
            return
        F = float(bar.close)
        if self._pnl_exits(F):
            return
        self._hedge(F, t)

    def _update_rv(self, iid, t, px):
        if not (time(9, 30) <= t.time() <= time(16, 0)) or t.minute % 5:
            return
        if self.rv_day is None or self.rv_day[0] != t.date():
            if self.rv_day is not None and self.rv_day[2] >= 30:
                self.rv_days.append(tuple(self.rv_day))
            self.rv_day = [t.date(), 0.0, 0]
            self.rv_last = None                          # no overnight return in RTH RV
        if self.rv_last is not None and self.rv_last[0] == iid and px > 0:
            r2 = math.log(px / self.rv_last[1]) ** 2
            self.rv_day[1] += r2
            self.rv_day[2] += 1
            if self.book and not self.book["exiting"] and t.date() >= self.first_day:
                self.book["rv_r2"] += r2
                self.book["rv_n"] += 1
        self.rv_last = (iid, px)

    def _pos(self, iid):
        return int(round(float(self.portfolio.net_position(iid))))

    def _leg_greeks(self, F, now):
        """[(iid, qty, delta, gamma, mid)] using each leg's current implied vol."""
        b = self.book
        T = self._T(b["exp"], now)
        df = math.exp(-self.c.rate * T)
        out = []
        for inst, is_call in ((b["put"], False), (b["call"], True)):
            K = float(inst.strike_price)
            q = self._quote(inst.id, self.c.stale_minutes * NS_MIN)
            mid = 0.5 * sum(q) if q else float("nan")
            iv = float(implied_vol(mid, F, K, T, is_call, df)) if q else float("nan")
            if not np.isfinite(iv):
                iv = b["iv_entry"]
            out.append((inst.id, self._pos(inst.id), float(b76_delta(F, K, T, iv, is_call, df)),
                        float(b76_gamma(F, K, T, iv, df)), mid))
        return out

    def _hedge(self, F, t):
        b = self.book
        greeks = self._leg_greeks(F, t)
        hedge_inst = self.cache.instrument(b["hedge_id"])
        ratio = float(b["put"].multiplier) / float(hedge_inst.multiplier)
        delta = sum(q * d for _, q, d, _, _ in greeks) * ratio + self._pos(b["hedge_id"])
        b["last_delta"] = delta
        if abs(delta) < self.c.hedge_band:
            return
        qty = int(round(delta))
        if qty == 0:
            return
        b["hedges"] += 1
        b["hedge_qty"] += abs(qty)
        self.day["hedges"] += 1
        self._submit(b["hedge_id"], OrderSide.SELL if qty > 0 else OrderSide.BUY, abs(qty), "hedge")

    def _cycle_pnl(self, F):
        """Mark-to-market P&L of the open cycle in USD (options at mid, hedge at F), net of fees."""
        b = self.book
        mult = float(b["put"].multiplier)
        val = b["opt_cash"] + b["hedge_cash"] - b["fees"]
        for inst in (b["put"], b["call"]):
            q = self._quote(inst.id)
            if q is None:
                return float("nan")
            val += self._pos(inst.id) * 0.5 * sum(q) * mult
        val += self._pos(b["hedge_id"]) * F * float(self.cache.instrument(b["hedge_id"]).multiplier)
        return val

    def _pnl_exits(self, F):
        b = self.book
        if self.c.take_profit <= 0 and self.c.stop_loss <= 0:
            return False
        pnl = self._cycle_pnl(F)
        if not np.isfinite(pnl):
            return False
        prem_usd = b["premium"] * b["n"] * float(b["put"].multiplier)
        if self.c.take_profit > 0 and pnl >= self.c.take_profit * prem_usd:
            self._exit("take_profit")
            return True
        if self.c.stop_loss > 0 and pnl <= -self.c.stop_loss * prem_usd:
            self._exit("stop_loss")
            return True
        return False

    # ------------------------------------------------------------------ orders
    def _submit(self, iid, side, qty, tag):
        inst = self.cache.instrument(iid)
        o = self.order_factory.market(iid, side, inst.make_qty(qty), tags=[tag])
        if self.book is not None:
            self.book["pending"] += 1
        self.submit_order(o)

    def _exit(self, reason):
        b = self.book
        b["exiting"], b["reason"] = True, reason
        b["closed"] = self._ny()
        F = self._price(b["hedge_id"])
        b["F_exit"] = F
        b["pnl_mark"] = self._cycle_pnl(F) if np.isfinite(F) else float("nan")
        sent = False
        for iid in (b["put"].id, b["call"].id, b["hedge_id"]):
            q = self._pos(iid)
            if q:
                self._submit(iid, OrderSide.SELL if q > 0 else OrderSide.BUY, abs(q), f"close:{reason}")
                sent = True
        if not sent:
            self._finish()

    def on_order_filled(self, e: OrderFilled):
        b = self.book
        if b is None:
            return
        iid = e.instrument_id
        if iid not in (b["put"].id, b["call"].id, b["hedge_id"]):
            return
        mult = float(self.cache.instrument(iid).multiplier)
        signed = float(e.last_qty) * (1 if e.order_side == OrderSide.BUY else -1)
        flow = -signed * float(e.last_px) * mult
        if iid == b["hedge_id"]:
            b["hedge_cash"] += flow
        else:
            b["opt_cash"] += flow
            book_px = b["opt_exit_px"] if b["exiting"] else b["opt_entry_px"]
            book_px[str(iid)] = float(e.last_px)
        if e.commission is not None:
            b["fees"] += float(e.commission.as_double())
        o = self.cache.order(e.client_order_id)
        if o is None or o.is_closed:
            b["pending"] = max(0, b["pending"] - 1)
        if b["exiting"] and b["pending"] == 0 and all(self._pos(i) == 0 for i in
                                                      (b["put"].id, b["call"].id, b["hedge_id"])):
            self._finish()

    def _order_failed(self, e):
        b = self.book
        if b is None:
            return
        b["pending"] = max(0, b["pending"] - 1)
        self.log.warning(f"order failed {e.instrument_id}: {getattr(e, 'reason', '')}")
        if not b["exiting"]:
            self._exit("order_failed")
        elif b["pending"] == 0 and all(self._pos(i) == 0 for i in (b["put"].id, b["call"].id, b["hedge_id"])):
            self._finish()

    def on_order_rejected(self, e: OrderRejected):
        self._order_failed(e)

    def on_order_denied(self, e: OrderDenied):
        self._order_failed(e)

    def _finish(self):
        b = self.book
        pnl = b["opt_cash"] + b["hedge_cash"] - b["fees"]
        mult = float(b["put"].multiplier)
        ex = b["opt_exit_px"]
        exit_prem = (ex.get(str(b["put"].id), np.nan) + ex.get(str(b["call"].id), np.nan))
        en = b["opt_entry_px"]
        entry_prem = en.get(str(b["put"].id), np.nan) + en.get(str(b["call"].id), np.nan)
        rv_hold = ann_vol(b["rv_r2"], b["rv_n"], self.c.rth_var_share)
        kind = ("long" if b["dir"] > 0 else "short") + ("_straddle" if b["K_put"] == b["K_call"] else "_strangle")
        row = dict(kind=kind, opened=b["opened"], closed=b.get("closed", self._ny()), reason=b["reason"],
                   n=b["n"], put=b["put"].raw_symbol.value, call=b["call"].raw_symbol.value,
                   expiry=b["exp"], dte_entry=(b["exp"] - b["opened"].date()).days,
                   F_entry=b["F_entry"], F_exit=b.get("F_exit"),
                   premium_entry=entry_prem, premium_exit=exit_prem,
                   iv_entry=b["iv_entry"], atm_iv=b["atm_iv"], rv_entry=b["rv_entry"], rv_hold=rv_hold,
                   regime=b["regime"], cushion=b["cushion"], flip=b["flip"],
                   call_wall=b["call_wall"], put_wall=b["put_wall"],
                   hedges=b["hedges"], hedge_contracts=b["hedge_qty"],
                   opt_pnl=b["opt_cash"], hedge_pnl=b["hedge_cash"], fees=b["fees"], pnl=pnl,
                   pnl_per_straddle=pnl / b["n"] / mult)
        self.trades.append(row)
        self.realized += pnl
        self.log.info(f"{kind} {b['opened']:%m-%d %H:%M} -> {row['closed']:%m-%d %H:%M} {b['reason']}: "
                      f"options ${b['opt_cash']:,.0f} hedges ${b['hedge_cash']:,.0f} ({b['hedges']} trades) "
                      f"fees ${b['fees']:,.0f} = ${pnl:,.0f}; iv {b['iv_entry']:.1%} vs rv {rv_hold:.1%}")
        self.book = None

    # ------------------------------------------------------------------ daily log
    def _new_day(self, d):
        if self.day is not None and self.day["date"] >= self.first_day:
            self._log_day()
        if self.start_equity is None and d is not None and d >= self.first_day:
            self.start_equity = self._cash()
        self.day = dict(date=d, entries=0, hedges=0)

    def _log_day(self):
        s = self.sig or {}
        b = self.book
        self.daily.append(dict(date=self.day["date"], equity=self._equity(), regime=s.get("regime"),
                               cushion=s.get("cushion"), flip=s.get("flip"), call_wall=s.get("call_wall"),
                               put_wall=s.get("put_wall"), gex_bn_per_1pct=(s.get("gex") or np.nan) / 1e9,
                               atm_iv=s.get("atm_iv"), rv=self._rv(),
                               position=("long" if b["dir"] > 0 else "short") if b else "",
                               net_delta=b.get("last_delta") if b else np.nan,
                               entries=self.day["entries"], hedges=self.day["hedges"]))

    def on_stop(self):
        if self.book and not self.book["exiting"]:
            self.log.warning("backtest ended with an open position; marked, not closed")
        if self.day is not None and self.day["date"] >= self.first_day:
            self._log_day()
        t = pd.DataFrame(self.trades)
        eq = self._equity()
        stats = {"Total P&L (incl. fees)": f"{eq - float(self.start_equity or eq):,.0f}",
                 "Cycles": len(t)}
        if self.book is not None:
            stats["Open at end (marked)"] = f"{self.book.get('last_mark', float('nan')):,.0f}"
        if len(t):
            for kind, g in t.groupby("kind"):
                stats[f"{kind} cycles"] = len(g)
                stats[f"{kind} win rate"] = f"{(g['pnl'] > 0).mean():.0%}"
                stats[f"{kind} P&L"] = f"{g['pnl'].sum():,.0f}"
                stats[f"{kind} options / hedges / fees"] = \
                    f"{g['opt_pnl'].sum():,.0f} / {g['hedge_pnl'].sum():,.0f} / {-g['fees'].sum():,.0f}"
                stats[f"{kind} mean IV paid vs RV held"] = f"{g['iv_entry'].mean():.1%} vs {g['rv_hold'].mean():.1%}"
            for r, g in t.groupby("reason"):
                stats[f"Exit {r}"] = f"{len(g)} cycles, {g['pnl'].sum():,.0f}"
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
FIELDS = list(EsGammaHarvestConfig.__struct_fields__)
DEFAULTS = dict(zip(FIELDS, EsGammaHarvestConfig.__struct_defaults__))


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


def build_config(catalog_path, start, end, cash, fee, params, out_dir, venue="GLBX",
                 oi_path=None, chunk_size=None, log_level="INFO", bar_source="auto",
                 oi_source=None, oi_volume_days=5, warmup_days=None, **_ignored):
    from nautilus_trader.config import (BacktestDataConfig, BacktestEngineConfig,
                                        BacktestRunConfig, BacktestVenueConfig, CacheConfig,
                                        DataEngineConfig, ImportableFeeModelConfig,
                                        ImportableStrategyConfig, LoggingConfig)
    from nautilus_trader.model.data import QuoteTick, TradeTick
    from nautilus_trader.persistence.catalog import ParquetDataCatalog

    root = params.get("root", DEFAULTS["root"])
    futs, opts = select_instruments(ParquetDataCatalog(catalog_path), root, venue)
    if not futs or not opts:
        raise SystemExit(f"No {root} futures/options for venue {venue} in {catalog_path}")
    if bar_source == "auto":
        bar_source = "EXTERNAL" if has_external_bars(catalog_path, futs) else "INTERNAL"
    if warmup_days is None:   # calendar days of futures data to fill the RV window
        warmup_days = int(math.ceil(params.get("rv_days", DEFAULTS["rv_days"]) * 1.5)) + 3
    fut_start = (pd.Timestamp(start) - pd.Timedelta(days=warmup_days)).isoformat()
    opt_start = (pd.Timestamp(start) - pd.Timedelta(days=1)).isoformat()
    data_end = (pd.Timestamp(end) + pd.Timedelta(days=1)).isoformat()
    oi_source = oi_source or params.get("oi_source", DEFAULTS["oi_source"])
    params = dict(params, oi_source="unit" if oi_source == "unit" else "file")
    if oi_source == "volume":
        oi_path = os.path.join(catalog_path, f"volume_oi_{oi_volume_days}d_{start}_{end}.parquet")
        if not os.path.exists(oi_path):
            print(f"building volume proxy for open interest ({oi_volume_days} sessions) ...")
            build_volume_oi(catalog_path, opts, start, end, oi_volume_days, oi_path)
    elif oi_source == "unit":
        oi_path = ""
    elif not oi_path:
        cand = os.path.join(catalog_path, "open_interest.parquet")
        oi_path = cand if os.path.exists(cand) else ""
    strategy = ImportableStrategyConfig(
        strategy_path="es_gamma_harvest:EsGammaHarvest",
        config_path="es_gamma_harvest:EsGammaHarvestConfig",
        config=dict(params, venue=venue, bar_source=bar_source,
                    trade_start_date=str(pd.Timestamp(start).date()), oi_path=oi_path,
                    output_dir=out_dir))
    venue_cfg = BacktestVenueConfig(
        name=venue, oms_type="NETTING", account_type="MARGIN", base_currency="USD",
        starting_balances=[f"{cash:.2f} USD"],
        fee_model=ImportableFeeModelConfig(
            fee_model_path="nautilus_trader.backtest.models:PerContractFeeModel",
            config_path="nautilus_trader.backtest.config:PerContractFeeModelConfig",
            config={"commission": f"{fee:.2f} USD"}))
    fut_ids = [str(f.id) for f in futs]
    data = [BacktestDataConfig(catalog_path=catalog_path, data_cls=QuoteTick, instrument_ids=fut_ids,
                               start_time=fut_start, end_time=data_end),
            BacktestDataConfig(catalog_path=catalog_path, data_cls=QuoteTick,
                               instrument_ids=[str(o.id) for o in opts],
                               start_time=opt_start, end_time=data_end)]
    if bar_source == "EXTERNAL":
        data.append(BacktestDataConfig(catalog_path=catalog_path, data_cls=Bar,
                                       bar_types=[f"{f}-1-MINUTE-LAST-EXTERNAL" for f in fut_ids],
                                       start_time=fut_start, end_time=data_end))
    else:
        data.append(BacktestDataConfig(catalog_path=catalog_path, data_cls=TradeTick, instrument_ids=fut_ids,
                                       start_time=fut_start, end_time=data_end))
    cfg = BacktestRunConfig(
        venues=[venue_cfg], data=data,
        engine=BacktestEngineConfig(
            strategies=[strategy], logging=LoggingConfig(log_level=log_level),
            cache=CacheConfig(tick_capacity=10, bar_capacity=10),
            data_engine=DataEngineConfig(time_bars_build_with_no_updates=False)),
        chunk_size=chunk_size, raise_exception=True, dispose_on_completion=False)
    info = dict(futures=sorted(f.raw_symbol.value for f in futs), options=len(opts),
                expiries=sorted({pd.Timestamp(o.expiration_ns, tz="UTC").date() for o in opts}),
                bar_source=bar_source, oi_path=oi_path, oi_source=oi_source, data_start=fut_start)
    return cfg, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--start", default="2026-01-12")
    ap.add_argument("--end", default="2026-03-13")
    ap.add_argument("--warmup-days", type=int, default=None,
                    help="calendar days of futures data before --start for RV (default: from rv-days)")
    ap.add_argument("--cash", type=float, default=1_000_000.0)
    ap.add_argument("--fee", type=float, default=2.0, help="commission per contract (futures and options), USD")
    ap.add_argument("--param", action="append", help="key=value (dashes ok), repeatable")
    ap.add_argument("--params-json", default=None)
    ap.add_argument("--venue", default="GLBX")
    ap.add_argument("--oi", default=None, help="open-interest parquet (default <catalog>/open_interest.parquet)")
    ap.add_argument("--oi-source", default=None, choices=["file", "volume", "unit"],
                    help="file: open_interest.parquet; volume: proxy from option trades; unit: weight 1")
    ap.add_argument("--oi-volume-days", type=int, default=5)
    ap.add_argument("--bars", default="auto", choices=["auto", "EXTERNAL", "INTERNAL"])
    ap.add_argument("--chunk-size", type=int, default=250_000)
    ap.add_argument("--out", default="results_gamma")
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args()

    from nautilus_trader.backtest.node import BacktestNode

    params, _ = parse_params(a.param, a.params_json)
    os.makedirs(a.out, exist_ok=True)
    cfg, info = build_config(a.catalog, a.start, a.end, a.cash, a.fee, params, a.out, venue=a.venue,
                             oi_path=a.oi, chunk_size=a.chunk_size, log_level=a.log_level,
                             bar_source=a.bars, oi_source=a.oi_source, oi_volume_days=a.oi_volume_days,
                             warmup_days=a.warmup_days)
    print(f"futures {info['futures']}, {info['options']} options on {len(info['expiries'])} expiries "
          f"({info['expiries'][0]} .. {info['expiries'][-1]}), bars {info['bar_source']}, "
          f"futures data from {info['data_start'][:10]}")
    print(f"GEX weights: {info['oi_source']}" + (f" ({info['oi_path']})" if info["oi_path"] else ""))
    if info["oi_source"] == "file" and not info["oi_path"]:
        print("WARNING: no open interest -> no GEX regime -> no trades. Use --oi-source volume "
              "or build open_interest.parquet with es_gex_futures.py --build-oi.")
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
