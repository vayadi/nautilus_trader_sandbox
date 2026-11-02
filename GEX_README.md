# ES gamma-exposure strategy: how to run it

Files in this folder:

| file | what it is |
|---|---|
| `es_gex_futures.py` | The strategy and backtest runner, plus `--build-oi` to turn Databento statistics into open interest |
| `optimize.py` | Parallel parameter grid, one backtest per process |
| `gex_grid.json` | Starter grid: 81 combinations of the core parameters |
| `thin_catalog.py` | Slim copy of the catalog for the GEX backtest: same results, less memory |
| `future_options_downloader_for_nautilus.py` | Your Databento → catalog downloader |

All commands run from `~/nautilus_trader_sandbox` in the `nautilus` conda environment.

## 1. Fix the data first

The catalog has three problems:

- **The weekly download stopped partway.** It covers 53 of 77 days, through 6 March; 1, 3 and 5 March and everything from 8 March on are missing. That's why there's no ES data in March.
- **The weekly options and ESM6 are invisible to the backtest.** Their quotes are in the catalog, but their instrument definitions aren't, because `convert` writes instruments only once.
- **The first and second downloads overlap.** New ESH6 quotes from 19 January on weren't written, because they clash with the files from the first conversion.

The fix: set the first-download files aside, finish the download with **exactly the same settings**, and rebuild the catalog. Commands are in steps 1a–1f below.

The settings matter because the downloader names files `<day>_<chunk>` and skips files that already exist. Your current `plan.parquet` comes from a later run with different settings: it splits 29 of the downloaded days into 3 files instead of 2. Resuming with those settings would mix two selections, so `check_plan.py` checks this before you spend anything.

```bash
conda activate nautilus
cd ~/nautilus_trader_sandbox

# 1a. find the exact command that made the _0/_1 files (around 15:26 on 31 Oct)
grep future_options_downloader ~/.zsh_history | tail -20

# 1b. set the first-download files and the old catalog aside (nothing is deleted)
cd fop_data
mkdir -p raw_first_download/bbo-1m raw_first_download/trades
for s in bbo-1m trades; do
  for f in raw/$s/2026-*.dbn.zst; do
    case "$f" in *_[0-9]*.dbn.zst) ;; *) mv "$f" raw_first_download/$s/ ;; esac
  done
done
mv catalog catalog_old && mv converted.json converted_old.json
cd ..

# 1c. dry run with that command (only --dry-run added): prints the cost of the missing files
#     and rewrites plan.parquet. Example with the defaults (dte-max 60, moneyness 0.10):
python future_options_downloader_for_nautilus.py download \
    --parents ES.OPT EW.OPT EW1.OPT EW2.OPT EW3.OPT EW4.OPT LO.OPT \
    --futures-parent ES.FUT CL.FUT \
    --start 2026-01-01 --end 2026-03-31 --schemas bbo-1m trades --dry-run

# 1d. safe to resume?   OK -> continue;   STOP -> try other settings from your history
python check_plan.py

# 1e. resume (same command, without --dry-run), then check again
python future_options_downloader_for_nautilus.py download \
    --parents ES.OPT EW.OPT EW1.OPT EW2.OPT EW3.OPT EW4.OPT LO.OPT \
    --futures-parent ES.FUT CL.FUT \
    --start 2026-01-01 --end 2026-03-31 --schemas bbo-1m trades
python check_plan.py                     # should now say 0 days still to download

# 1f. rebuild the catalog (instruments included this time)
python future_options_downloader_for_nautilus.py convert --schemas bbo-1m trades
```

**If 1d says STOP and you can't find the original command,** download into a fresh folder instead. That's clean but repays the 53 days you already have: add `--out ./fop_data_v2` to the download and convert commands in 1c, 1e and 1f, skip 1d, and use `fop_data_v2/catalog` everywhere below. You can move the files from 1b back where they were if you like.

`check_plan.py` catches any change that alters how days split into files. It can't detect a small change that keeps the same split, so the command from your history is the reliable reference.

The first line of every backtest shows what it found. After the fix it should list `ESH6` and `ESM6` and around 40 expiries, not 1:

```
futures ['ESH6', 'ESM6'], N options on M expiries (2026-01-02 .. ), bars INTERNAL
```

## 2. Open interest (optional for now)

Without open interest, use `--oi-source volume`: the last 5 sessions of option volume. With real open interest:

```bash
python future_options_downloader_for_nautilus.py download \
    --parents ES.OPT EW.OPT EW1.OPT EW2.OPT EW3.OPT EW4.OPT LO.OPT \
    --futures-parent ES.FUT CL.FUT \
    --start 2026-01-01 --end 2026-03-31 --schemas statistics --no-underlying
python es_gex_futures.py --catalog fop_data/catalog --build-oi fop_data/raw/statistics
# then run with the default --oi-source file
```

## 3. Single backtest

```bash
python es_gex_futures.py --catalog fop_data/catalog --start 2026-01-02 --end 2026-03-19 \
    --oi-source volume --out results_gex
```

ESH6 options expire on 20 March, and the weeklies run past that, but check the expiry range on the first line. Results go to `results_gex/`: `trades.csv`, `signals.csv` (one row per session), `summary.csv`, `fills.csv`, `account.csv`.

## 4. Optimization

```bash
python optimize.py --module es_gex_futures --catalog fop_data/catalog \
    --start 2026-01-02 --end 2026-02-27 --grid gex_grid.json \
    --fixed oi-source=volume --fixed contracts=1 \
    --workers 6 --target sharpe --out opt_gex
```

- **Hold out March.** Pick settings from the January to February results, then run that one set once on March (`--start 2026-03-02 --end 2026-03-19`) without changing it.
- **Workers.** Each worker holds one backtest in memory. Start with fewer workers than cores and watch RAM.
- **Reading the results.** `opt_gex/optimization.csv` is ranked by `--target`. Prefer a plateau of good neighbours to one isolated best combination, and ignore rows with only a handful of trades.
- **Testing each half.** Add `--fixed mom-enabled=0` or `--fixed mr-enabled=0` to test the two halves of the strategy separately.

## Memory

1. **Slim catalog (biggest saving).** The strategy only reads option quotes at its 15-minute rebalances during 09:30–15:45 ET, so the other ~95% of 1-minute option quotes can go. Build the copy once after each download (it uses hard links for everything else, so it takes little disk), then point `--catalog` at it:

   ```bash
   python thin_catalog.py fop_data/catalog fop_data/catalog_gex
   python es_gex_futures.py --catalog fop_data/catalog_gex --start 2026-01-02 --end 2026-03-19 --oi-source volume --out results_gex
   ```

   Results are identical to the full catalog, provided `rebalance-minutes` divides 15 and `stale-minutes` ≤ 30. Rebuild the copy if you change those.
2. **`--chunk-size`** (default now 250,000) sets how many records are loaded at once. Lower it, e.g. to 100,000, if memory is still tight. It doesn't change the results.
3. **Quote cache.** Nautilus keeps only the last 10 quotes per instrument now, instead of 10,000. This is built in.
4. **Optimizer.** Each worker is a full backtest, so total memory ≈ workers × one run. Measure one run first, then set `--workers` to what fits in your RAM.

## Parameters

| core (in the grid) | default | |
|---|---|---|
| `gex-buffer` | 0.0025 | distance from the flip, as a fraction of spot, needed to call a regime |
| `z-entry` / `z-stop` | 1.0 / 2.0 | VWAP-fade entry and stop, in expected-move units |
| `mom-stop-sigma` | 0.5 | breakout stop and trail, in daily σ |
| `or-minutes` | 30 | opening-range length |

Secondary: `z-exit`, `mom-use-walls`, `exit-on-regime-change`, `trade-start`, `last-entry`, `max-trades-day`, `rebalance-minutes`, `mr-enabled`, `mom-enabled`. Everything else is modelling choices; leave those fixed.
