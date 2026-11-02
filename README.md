# Futures-options downloader for NautilusTrader

Downloads CME futures options (default: ES on `GLBX.MDP3`) from Databento and writes them into a
Nautilus `ParquetDataCatalog`: `OptionContract` + `FuturesContract` instruments, plus quotes/trades/bars.

```bash
pip install -U nautilus_trader databento
export DATABENTO_API_KEY=db-...
```

## 1. Find the option groups (weeklies) on ES

`ES.OPT` only covers the standard quarterly/serial options. Weeklies and dailies live under other
product codes (EW, EW1–4, E1A…, etc.). `discover` pulls one day of all GLBX definitions and lists every
option group whose underlying is an ES future:

```bash
python future_options_downloader_for_nautilus.py discover --root ES --date 2026-10-30
# -> Use as --parents: ES.OPT EW.OPT E1A.OPT ...
```

## 2. Plan + cost check (no data spend)

```bash
python future_options_downloader_for_nautilus.py download --parents ES.OPT EW.OPT EW1.OPT EW2.OPT EW3.OPT EW4.OPT \
  --start 2025-01-01 --end 2025-12-31 --dte-max 60 --moneyness 0.10 \
  --schemas bbo-1m trades --dry-run
```

The selection is done per trading day: contracts alive that day, with `dte-min <= DTE <= dte-max`,
and `|K/F - 1| <= moneyness` against the prior daily close of the option's *own* underlying future.
Pass `--moneyness -1` to take the whole chain. The underlying futures are fetched in the same schemas
unless `--no-underlying`.

## 3. Download + convert

```bash
python future_options_downloader_for_nautilus.py run  ...same args...  --max-cost 50      # prompts before spending
python future_options_downloader_for_nautilus.py convert --schemas bbo-1m trades          # (re)convert only
```

Layout under `--out` (default `./fop_data`):

```
raw/definition/*.dbn.zst      monthly definitions per parent
raw/ohlcv-1d/                 futures daily bars (reference prices)
raw/<schema>/<YYYY-MM-DD>.dbn.zst
plan.parquet                  (day, raw_symbol, kind) selection
catalog/                      Nautilus ParquetDataCatalog
converted.json                conversion manifest
```

Both steps are resumable: existing DBN files are skipped, and files already in `converted.json` are not
written to the catalog twice. Delete `converted.json` + `catalog/` to rebuild the catalog from raw.

## Schema notes

| schema   | Nautilus type            | comment |
|----------|--------------------------|---------|
| `bbo-1m` / `bbo-1s` | `QuoteTick`   | sampled top of book – best default for options |
| `trades` | `TradeTick`              | sparse for OTM strikes |
| `tbbo` / `mbp-1` | `QuoteTick` (+ `TradeTick` with `--include-trades`) | tick-level, much pricier |
| `ohlcv-1m` | `Bar`                  | trade-based bars, gaps where no trades |

Use in a backtest:

```python
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.backtest.node import BacktestDataConfig
from nautilus_trader.model.data import QuoteTick

catalog = ParquetDataCatalog("./fop_data/catalog")
data_cfg = BacktestDataConfig(catalog_path=str(catalog.path), data_cls=QuoteTick,
                              start_time="2025-01-01", end_time="2025-12-31")
```
