#!/usr/bin/env python3
"""
Consolidate the Nautilus Parquet catalog written by future_options_downloader_for_nautilus.py.

Why
---
The downloader converts one Databento file per schema per trading day, and Nautilus writes
one Parquet file per instrument per day for it. A year of ES options therefore ends up as
tens of thousands of small files, which makes catalog queries and BacktestNode loading slow.
Consolidation merges each instrument's daily files into one file per data type.

When
----
Run it *after* `run` / `convert` has finished. If you later add data (more dates, weeklies,
another product such as LO/CL), just run this again afterwards; the new daily files get
merged in the same way.

Usage
-----
    python consolidate_catalog.py                       # ./fop_data/catalog
    python consolidate_catalog.py --catalog /path/to/catalog
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nautilus_trader.persistence.catalog import ParquetDataCatalog


def count_parquet_files(path: Path) -> int:
    return sum(1 for _ in path.rglob("*.parquet"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Consolidate a Nautilus Parquet catalog")
    ap.add_argument("--catalog", default="./fop_data/catalog",
                    help="catalog directory (the downloader's --out + /catalog)")
    args = ap.parse_args()

    path = Path(args.catalog)
    if not path.is_dir():
        raise SystemExit(f"Catalog not found: {path} (did the download/convert step run?)")

    catalog = ParquetDataCatalog(path)
    before = count_parquet_files(path)

    # ensure_contiguous_files=False is required here: the downloader writes one file per
    # trading day, so weekends/holidays (and days on which a contract wasn't selected
    # because it was outside the DTE / moneyness window) leave time gaps between files.
    # With the default (True) Nautilus asserts the files are back-to-back and fails.
    #
    # deduplicate is left at False: the downloader's converted.json manifest already stops
    # the same DBN file from being written twice, so the data has no duplicates to remove.
    catalog.consolidate_catalog(ensure_contiguous_files=False)

    after = count_parquet_files(path)
    print(f"Parquet files: {before} -> {after}")

    # Quick sanity check that everything is still readable after the merge.
    print(f"Instruments: {len(catalog.instruments())}")
    print(f"Data types:  {', '.join(catalog.list_data_types())}")


if __name__ == "__main__":
    main()
