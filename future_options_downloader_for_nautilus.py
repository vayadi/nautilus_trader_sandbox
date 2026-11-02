#!/usr/bin/env python3
"""
Futures-options data downloader for NautilusTrader (Databento -> Nautilus Parquet catalog).

Pipeline
--------
1. Download option + underlying-futures definitions (parent symbology, e.g. ES.OPT / ES.FUT).
2. Download daily futures bars (ohlcv-1d) to get a reference price per underlying.
3. For every trading day, select the option contracts that are alive, inside a DTE window
   and inside a moneyness band around their own underlying future's prior close.
4. Estimate cost, ask for confirmation, then download the requested schemas per day
   (raw DBN files, resumable: existing files are skipped).
5. Convert everything into a Nautilus ParquetDataCatalog with DatabentoDataLoader
   (instruments first, so price precisions are known; idempotent via a manifest).

Subcommands
-----------
  discover   List the option product groups (weeklies etc.) that sit on a futures root.
  download   Steps 1-4 (use --dry-run to stop after the cost estimate).
  convert    Step 5.
  run        download + convert.

Requires: pip install nautilus_trader databento ; env DATABENTO_API_KEY.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

log = logging.getLogger("fop")

CHUNK = 2000  # max symbols per Databento request
MONTH_CODES = "FGHJKMNQUVXZ"


# --------------------------------------------------------------------------------------
# Config / layout
# --------------------------------------------------------------------------------------


@dataclass
class Layout:
    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def catalog(self) -> Path:
        return self.root / "catalog"

    @property
    def plan_file(self) -> Path:
        return self.root / "plan.parquet"

    @property
    def manifest(self) -> Path:
        return self.root / "converted.json"

    def defs_dir(self) -> Path:
        return self.raw / "definition"

    def schema_dir(self, schema: str) -> Path:
        return self.raw / schema


def _client():
    import databento as db

    if not os.environ.get("DATABENTO_API_KEY"):
        sys.exit("DATABENTO_API_KEY is not set")
    return db.Historical()


def _d(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _month_windows(start: date, end_excl: date):
    cur = start
    while cur < end_excl:
        nxt = (pd.Timestamp(cur) + pd.offsets.MonthBegin(1)).date()
        yield cur, min(nxt, end_excl)
        cur = nxt


def _fetch(client, path: Path, **kw) -> Path:
    """Download to a temp file and rename atomically (makes reruns safe)."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    client.timeseries.get_range(path=tmp, **kw)
    os.replace(tmp, path)
    log.info("saved %s", path)
    return path


def _confirm(total: float, max_cost: float | None, yes: bool) -> None:
    print(f"\nEstimated Databento cost: ${total:,.2f}")
    if max_cost is not None and total > max_cost:
        sys.exit(f"Aborting: estimate exceeds --max-cost ${max_cost:,.2f}")
    if not yes and total > 0:
        if input("Proceed? [y/N] ").strip().lower() != "y":
            sys.exit("Aborted")


# --------------------------------------------------------------------------------------
# discover
# --------------------------------------------------------------------------------------


def cmd_discover(a) -> None:
    """One day of ALL_SYMBOLS definitions -> option groups whose underlying is <root> futures."""
    client = _client()
    day = _d(a.date)
    kw = dict(
        dataset=a.dataset,
        start=day.isoformat(),
        end=(day + timedelta(days=1)).isoformat(),
        symbols="ALL_SYMBOLS",
        schema="definition",
    )
    cost = client.metadata.get_cost(**kw)
    _confirm(cost, a.max_cost, a.yes)
    path = _fetch(client, Layout(Path(a.out)).defs_dir() / f"ALL_{day}.dbn.zst", **kw)

    import databento as db

    df = db.DBNStore.from_file(path).to_df(map_symbols=False)
    fut = re.compile(rf"^{re.escape(a.root)}[{MONTH_CODES}]\d{{1,2}}$")
    opts = df[df["instrument_class"].isin(["C", "P"]) & df["underlying"].str.match(fut)]
    if opts.empty:
        print(f"No options on {a.root} futures found for {day}")
        return
    summary = (
        opts.drop_duplicates("raw_symbol")
        .groupby("asset")
        .agg(n=("raw_symbol", "size"), example=("raw_symbol", "first"),
             next_expiry=("expiration", "min"))
        .sort_values("next_expiry")
    )
    print(summary.to_string())
    print("\nUse as --parents:", " ".join(f"{x}.OPT" for x in summary.index))


# --------------------------------------------------------------------------------------
# download
# --------------------------------------------------------------------------------------


def download_definitions(client, a, lay: Layout, start: date, end: date) -> list[Path]:
    parents = list(a.parents) + list(a.futures_parent)
    paths = []
    for ws, we in _month_windows(start, end):
        for p in parents:
            paths.append(
                _fetch(
                    client,
                    lay.defs_dir() / f"{p}_{ws:%Y-%m}.dbn.zst",
                    dataset=a.dataset,
                    start=ws.isoformat(),
                    end=we.isoformat(),
                    symbols=[p],
                    stype_in="parent",
                    schema="definition",
                ),
            )
    return paths


def load_definitions(paths: list[Path]) -> pd.DataFrame:
    import databento as db

    frames = [db.DBNStore.from_file(p).to_df(map_symbols=False) for p in paths if p.stat().st_size]
    df = pd.concat(frames)
    df = df.sort_index().drop_duplicates("instrument_id", keep="last")
    cols = ["instrument_id", "raw_symbol", "instrument_class", "asset", "underlying",
            "underlying_id", "strike_price", "expiration", "activation"]
    df = df[[c for c in cols if c in df.columns]].reset_index(drop=True)
    # Normalise dtypes: pandas 3 / pyarrow string defaults otherwise leak into arithmetic.
    df["strike_price"] = pd.to_numeric(df["strike_price"], errors="coerce").astype("float64")
    df["instrument_id"] = df["instrument_id"].astype("int64")
    if "underlying_id" in df:
        df["underlying_id"] = pd.to_numeric(df["underlying_id"], errors="coerce").fillna(0).astype("int64")
    for c in ("raw_symbol", "underlying", "instrument_class", "asset"):
        df[c] = df[c].astype("object")
    return df


def download_underlying_daily(client, a, lay: Layout, start: date, end: date,
                              defs: pd.DataFrame) -> pd.DataFrame:
    """Daily futures closes with columns day, instrument_id, symbol, close.

    Parent-symbology files map every row to the parent ("ES.FUT"), so raw symbols
    come from the definitions via instrument_id instead.
    """
    import databento as db

    frames = []
    for fp in a.futures_parent:
        path = _fetch(
            client,
            lay.raw / "ohlcv-1d" / f"{fp}_{start}_{end}.dbn.zst",
            dataset=a.dataset,
            start=(start - timedelta(days=10)).isoformat(),
            end=end.isoformat(),
            symbols=[fp],
            stype_in="parent",
            schema="ohlcv-1d",
        )
        if path.stat().st_size:
            frames.append(db.DBNStore.from_file(path).to_df(map_symbols=False).reset_index())
    bars = pd.concat(frames, ignore_index=True)
    bars["instrument_id"] = bars["instrument_id"].astype("int64")
    bars["close"] = pd.to_numeric(bars["close"], errors="coerce").astype("float64")
    bars["symbol"] = bars["instrument_id"].map(
        defs.set_index("instrument_id")["raw_symbol"]).astype("object")
    bars["day"] = bars["ts_event"].dt.tz_convert("UTC").dt.date
    bars = bars.dropna(subset=["close"]).drop_duplicates(["day", "instrument_id"], keep="last")
    return bars[["day", "instrument_id", "symbol", "close"]]


def build_plan(a, defs: pd.DataFrame, closes: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Rows = (day, raw_symbol, kind) for everything we will request."""
    opts = defs[defs["instrument_class"].isin(["C", "P"])].copy()
    if a.rights != "both":
        opts = opts[opts["instrument_class"] == ("C" if a.rights == "calls" else "P")]
    opts["exp_day"] = opts["expiration"].dt.tz_convert("UTC").dt.date
    opts["act_day"] = opts["activation"].dt.tz_convert("UTC").dt.date

    close_days = sorted(closes["day"].unique())
    by_day = {d: g for d, g in closes.groupby("day")}
    unmatched_days = 0
    rows = []
    day = start
    while day < end:
        if day.weekday() != 5:  # Globex is shut on Saturdays
            prior = [d for d in close_days if d < day]
            alive = opts[(opts["act_day"] <= day) & (opts["exp_day"] >= day)]
            dte = alive["exp_day"].map(lambda e: (e - day).days)
            sel = alive[(dte >= a.dte_min) & (dte <= a.dte_max)]
            if a.moneyness is not None and prior:
                ref = by_day[prior[-1]]
                # Prefer the numeric underlying_id link; fall back to the underlying symbol.
                fwd = pd.Series(float("nan"), index=sel.index, dtype="float64")
                if "underlying_id" in sel:
                    fwd = sel["underlying_id"].map(ref.set_index("instrument_id")["close"])
                by_sym = ref.dropna(subset=["symbol"]).drop_duplicates("symbol", keep="last")
                fwd = pd.to_numeric(fwd, errors="coerce").astype("float64").fillna(
                    pd.to_numeric(sel["underlying"].map(by_sym.set_index("symbol")["close"]),
                                  errors="coerce").astype("float64"),
                )
                if len(sel) and fwd.isna().all():
                    unmatched_days += 1
                band = (sel["strike_price"].astype("float64") / fwd - 1).abs() <= a.moneyness
                sel = sel[band.fillna(False).astype(bool)]
            for sym in sel["raw_symbol"]:
                rows.append((day, sym, "option"))
            if a.include_underlying:
                for sym in sel["underlying"].dropna().unique():
                    rows.append((day, sym, "future"))
        day += timedelta(days=1)
    if unmatched_days:
        log.warning("%d days had no underlying close for any option; check --futures-parent "
                    "(e.g. sample underlying=%r)", unmatched_days,
                    opts["underlying"].dropna().head(3).tolist())
    plan = pd.DataFrame(rows, columns=["day", "raw_symbol", "kind"]).drop_duplicates()
    return plan


def _requests(a, lay: Layout, plan: pd.DataFrame):
    """Yield (path, kwargs) per (schema, day, chunk)."""
    for schema in a.schemas:
        for day, grp in plan.groupby("day"):
            syms = sorted(grp["raw_symbol"])
            for i in range(0, len(syms), CHUNK):
                part = f"_{i // CHUNK}" if len(syms) > CHUNK else ""
                yield (
                    lay.schema_dir(schema) / f"{day}{part}.dbn.zst",
                    dict(
                        dataset=a.dataset,
                        start=day.isoformat(),
                        end=(day + timedelta(days=1)).isoformat(),
                        symbols=syms[i : i + CHUNK],
                        stype_in="raw_symbol",
                        schema=schema,
                    ),
                )


def _check_parents(client, a, start: date, end: date) -> tuple[float, list[str]]:
    """Price each parent separately so an unknown symbol is reported by name."""
    from databento.common.error import BentoClientError

    total, bad = 0.0, []
    for p in [*a.parents, *a.futures_parent]:
        try:
            total += client.metadata.get_cost(
                dataset=a.dataset, start=start.isoformat(), end=end.isoformat(),
                symbols=[p], stype_in="parent", schema="definition",
            )
        except BentoClientError as e:
            if "symbology" not in str(e):
                raise
            bad.append(p)
    return total, bad


def cmd_download(a) -> None:
    lay = Layout(Path(a.out))
    client = _client()
    start, end = _d(a.start), _d(a.end) + timedelta(days=1)  # --end is inclusive

    # Metadata (definitions + daily bars) is cheap; still show it.
    meta_cost, bad = _check_parents(client, a, start, end)
    if bad:
        sys.exit(
            f"Databento could not resolve: {', '.join(bad)} on {a.dataset}.\n"
            "Option product codes often differ from the futures code (e.g. WTI crude options are "
            "LO.OPT, not CL.OPT; 10y note options are OZN.OPT). Find the right ones with:\n"
            "  python future_options_downloader_for_nautilus.py discover --root CL --date <trading day>",
        )
    print(f"Definitions cost estimate: ${meta_cost:,.2f}")
    defs = load_definitions(download_definitions(client, a, lay, start, end))
    closes = download_underlying_daily(client, a, lay, start, end, defs)

    plan = build_plan(a, defs, closes, start, end)
    if plan.empty:
        sys.exit("Selection is empty - widen --dte-max / --moneyness or check --parents")
    lay.root.mkdir(parents=True, exist_ok=True)
    plan.to_parquet(lay.plan_file)
    per_day = plan.groupby("day")["raw_symbol"].size()
    print(f"Plan: {plan['raw_symbol'].nunique()} contracts over {len(per_day)} days "
          f"(avg {per_day.mean():.0f}/day, max {per_day.max()})")

    todo = [(p, kw) for p, kw in _requests(a, lay, plan) if not p.exists()]
    print(f"Requests to make: {len(todo)} (schemas: {', '.join(a.schemas)})")
    if not todo:
        return
    with ThreadPoolExecutor(a.workers) as ex:
        costs = list(ex.map(lambda r: client.metadata.get_cost(**r[1]), todo))
    _confirm(sum(costs), a.max_cost, a.yes or a.dry_run)
    if a.dry_run:
        return

    def job(r):
        try:
            _fetch(client, r[0], **r[1])
        except Exception as e:  # keep going; rerun picks up the rest
            log.error("failed %s: %s", r[0], e)
            return r[0]

    with ThreadPoolExecutor(a.workers) as ex:
        failed = [x for x in ex.map(job, todo) if x]
    if failed:
        sys.exit(f"{len(failed)} requests failed - rerun the same command to resume")
    print("Download complete")


# --------------------------------------------------------------------------------------
# convert
# --------------------------------------------------------------------------------------


def cmd_convert(a) -> None:
    from nautilus_trader.adapters.databento import DatabentoDataLoader
    from nautilus_trader.persistence.catalog import ParquetDataCatalog

    lay = Layout(Path(a.out))
    lay.catalog.mkdir(parents=True, exist_ok=True)
    catalog = ParquetDataCatalog(lay.catalog)
    loader = DatabentoDataLoader()
    done = set(json.loads(lay.manifest.read_text())) if lay.manifest.exists() else set()

    wanted = None
    if lay.plan_file.exists():
        wanted = set(pd.read_parquet(lay.plan_file)["raw_symbol"])

    # Instruments: always (re)load so the loader's price-precision cache is filled.
    instruments = {}
    for p in sorted(lay.defs_dir().glob("*.dbn.zst")):
        if p.name.startswith("ALL_") or not p.stat().st_size:
            continue
        for inst in loader.from_dbn_file(p, as_legacy_cython=True, skip_on_error=True):
            instruments[inst.id] = inst  # later snapshots win
    if wanted is not None:
        instruments = {k: v for k, v in instruments.items() if v.raw_symbol.value in wanted}
    if "instruments" not in done:
        catalog.write_data(list(instruments.values()))
        done.add("instruments")
        print(f"wrote {len(instruments)} instruments")

    for schema in a.schemas:
        for p in sorted(lay.schema_dir(schema).glob("*.dbn.zst")):
            key = str(p.relative_to(lay.raw))
            if key in done or not p.stat().st_size:
                continue
            data = loader.from_dbn_file(
                p, as_legacy_cython=False, include_trades=a.include_trades,
            )
            if data:
                catalog.write_data(data)
            done.add(key)
            lay.manifest.write_text(json.dumps(sorted(done)))
            print(f"{key}: {len(data)} records")
    lay.manifest.write_text(json.dumps(sorted(done)))
    print(f"Catalog: {lay.catalog}")


def cmd_run(a) -> None:
    cmd_download(a)
    if not a.dry_run:
        cmd_convert(a)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--dataset", default="GLBX.MDP3")
        p.add_argument("--out", default="./fop_data", help="root dir for raw DBN + catalog")
        p.add_argument("--max-cost", type=float, help="abort if estimate (USD) exceeds this")
        p.add_argument("-y", "--yes", action="store_true", help="skip cost confirmation")

    d = sub.add_parser("discover", help="list option groups on a futures root")
    common(d)
    d.add_argument("--root", default="ES")
    d.add_argument("--date", required=True, help="YYYY-MM-DD (a trading day)")
    d.set_defaults(fn=cmd_discover)

    def dl(p):
        p.add_argument("--parents", nargs="+", default=["ES.OPT"],
                       help="option parent symbols, e.g. ES.OPT EW.OPT E1A.OPT (see `discover`)")
        p.add_argument("--futures-parent", nargs="+", default=["ES.FUT"],
                       help="parent(s) of the underlying futures, one per product "
                            "(e.g. ES.FUT CL.FUT when also using LO.OPT)")
        p.add_argument("--start", required=True, help="YYYY-MM-DD")
        p.add_argument("--end", required=True, help="YYYY-MM-DD, inclusive")
        p.add_argument("--dte-min", type=int, default=0)
        p.add_argument("--dte-max", type=int, default=60)
        p.add_argument("--moneyness", type=float, default=0.10,
                       help="keep |K/F-1| <= x vs prior close of the option's own future; "
                            "pass -1 to disable")
        p.add_argument("--rights", choices=["both", "calls", "puts"], default="both")
        p.add_argument("--no-underlying", dest="include_underlying", action="store_false",
                       help="don't fetch the same schemas for the underlying futures")
        p.add_argument("--workers", type=int, default=4)
        p.add_argument("--dry-run", action="store_true", help="stop after the cost estimate")

    def cv(p):
        p.add_argument("--schemas", nargs="+", default=["bbo-1m", "trades"],
                       help="e.g. bbo-1m bbo-1s trades tbbo mbp-1 ohlcv-1m cbbo-1m tcbbo")
        p.add_argument("--include-trades", action="store_true",
                       help="for mbp-1/tbbo/tcbbo also emit TradeTicks")

    for name, fn, parts in [("download", cmd_download, (dl, cv)), ("convert", cmd_convert, (cv,)),
                            ("run", cmd_run, (dl, cv))]:
        p = sub.add_parser(name)
        common(p)
        for f in parts:
            f(p)
        p.set_defaults(fn=fn)

    a = ap.parse_args(argv)
    if getattr(a, "moneyness", None) is not None and a.moneyness < 0:
        a.moneyness = None
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    a.fn(a)


if __name__ == "__main__":
    main()
