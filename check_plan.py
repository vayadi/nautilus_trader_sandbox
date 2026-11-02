"""Check that fop_data/plan.parquet matches the chunk files already downloaded.

The downloader names files <day>_<i>.dbn.zst after splitting each day's sorted symbol list
into chunks of 2000, and skips files that already exist. If the plan changed since those
files were downloaded, resuming would skip existing chunks that hold a different set of
symbols and fetch mismatched new ones. Run this right after a --dry-run, which rewrites
plan.parquet with the plan the next download would use.

  python check_plan.py            # exit code 0 = safe to resume, 1 = plan differs
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "fop_data")
CHUNK = 2000

plan = pd.read_parquet(ROOT / "plan.parquet")
plan["day"] = pd.to_datetime(plan["day"]).dt.date.astype(str)
per_day = plan.groupby("day")["raw_symbol"].size()
need = {d: (-(-n // CHUNK) if n > CHUNK else 1) for d, n in per_day.items()}

ok = True
for schema in ("bbo-1m", "trades"):
    d = ROOT / "raw" / schema
    if not d.is_dir():
        continue
    have = {}
    for f in d.glob("2026-*_[0-9]*.dbn.zst"):
        have[f.name[:10]] = have.get(f.name[:10], 0) + 1
    bad = {day: (n, need.get(day, 0)) for day, n in have.items() if need.get(day, 0) != n}
    missing = [day for day in need if day not in have]
    print(f"{schema}: {len(have)} days downloaded, {len(missing)} days still to download")
    if bad:
        ok = False
        sample = ", ".join(f"{k} has {v[0]} files, plan needs {v[1]}" for k, v in list(bad.items())[:4])
        print(f"  PLAN DIFFERS on {len(bad)} downloaded days, e.g. {sample}")

if ok:
    print("OK: the plan matches the files on disk, resuming is safe.")
else:
    print("STOP: these settings differ from the ones used for the files on disk.\n"
          "Find the original command (grep future_options_downloader ~/.zsh_history) or\n"
          "download into a fresh --out folder instead.")
sys.exit(0 if ok else 1)
