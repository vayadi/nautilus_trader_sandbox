# rates_VectorAutoRegression: interest-rate VAR (HUF + EUR, USD, JPY, GBP, CHF, AUD)

A VAR model of HUF interest rates, with EUR, USD, JPY, GBP, CHF and AUD rates as inputs.

```bash
pip install -U polars numpy scipy statsmodels matplotlib threadpoolctl yfinance databento
```

## 1. Download about 10 years of daily rates

```bash
python rates_VectorAutoRegression/rates_VectorAutoRegression_data.py --source official  --tenor 10y                  # recommended, all 7 incl. HUF (ÁKK 10Y)
python rates_VectorAutoRegression/rates_VectorAutoRegression_data.py --source official  --tenor short                # HUF = ÁKK 3M T-bill
python rates_VectorAutoRegression/rates_VectorAutoRegression_data.py --akk-dump                                      # all ÁKK tenors -> data/akk_benchmark_yields.csv
python rates_VectorAutoRegression/rates_VectorAutoRegression_data.py --source yfinance  --tenor short
python rates_VectorAutoRegression/rates_VectorAutoRegression_data.py --source databento --tenor short --dry-run      # cost check first
python rates_VectorAutoRegression/rates_VectorAutoRegression_data.py --source databento --tenor short --max-cost 5
```

This writes `rates_VectorAutoRegression/data/rates_<source>_<tenor>.parquet` (plus `.csv`, and a `.json` that maps each currency to its series).

Neither Yahoo nor Databento carries real interest-rate series for most of these currencies, so this is what each source uses:

| ccy | official 10y | official short | yfinance short | databento short (GLBX) |
|---|---|---|---|---|
| USD | FRED DGS10 | FRED DGS3MO | ^IRX 13w T-bill | ZQ fed funds futures, averaged over the FX-futures window |
| EUR | ECB AAA curve 10Y | ECB AAA curve 3M | CIP: EURUSD=X vs 6E=F | CIP: 6E front/second calendar spread |
| GBP | BoE IUDMNPY | BoE SONIA | CIP: GBPUSD=X vs 6B=F | CIP: 6B |
| JPY | MoF JGB 10Y | MoF JGB 1Y | CIP: JPY=X vs 6J=F | CIP: 6J |
| CHF | SNB rendeiduebd 10J (spot, published monthly) | SNB SARON | CIP: CHF=X vs 6S=F | CIP: 6S |
| AUD | RBA F2 10Y | RBA F1 90d bank bill | CIP: AUDUSD=X vs 6A=F | CIP: 6A |
| HUF | ÁKK reference yield 10Y | ÁKK reference yield 3M | not on Yahoo, uses --fallback (ÁKK 3M) | CIP on CME HUF/USD futures (`--huf-root`, thin); `--override HUF=akk` for ÁKK 3M |

"CIP" means the covered-interest-parity implied rate: `r_ccy = ((1 + r_usd*T) * F1/F2 - 1) / T`. On Databento this uses the two nearest quarterly futures, so it is a ~3M forward rate with no spot needed. On Yahoo it is spot against the front future, smoothed with `--smooth` (EWMA). The Yahoo version is noisy because spot and futures closes are not taken at the same time.

You can override any currency with `--override JPY=official` or `--csv CCY=file.csv` (a CSV wins over every source). The CSV loader accepts Hungarian formats such as `2025.01.02.` dates and comma decimals.

### HUF: ÁKK reference yields

HUF comes from the ÁKK (Hungarian Government Debt Management Agency) daily reference yields, the data behind akk.hu, Statisztika, Referenciahozamok. The page reads a public JSON API:

```
GET https://www.akk.hu/api/data/xpose3-dwh/data?query=getBenchmark&dateStart=2016-01-04&dateEnd=2026-11-24
```

It returns one row per fixing day and tenor, with fields such as `yield` (the reference yield in %, the mid of `avgBid`/`avgAsk`), `name`/`isin` of the benchmark paper, and `netPrice`. Tenors are 3M, 6M and 12M (DKJ T-bills) and 3Y, 5Y, 10Y, 15Y and 20Y (bonds). History starts on 2016-01-04; 20Y starts on 2020-05-20. There is no range limit (ten years is one ~6 MB call).

- `akk_yields(tenor, start, end)` in `rates_VectorAutoRegression_data.py` returns `(dates, yields)`. The raw JSON is cached per calendar year in `<out-dir>/raw/akk/benchmark_<year>.json`. Years that have ended are read from the cache and never fetched again; only the current year is re-fetched. `--akk-refresh` re-fetches every year.
- The official source maps `--tenor 10y` to ÁKK 10Y and `--tenor short` to ÁKK 3M. `--override HUF=akk` (or `--source akk`) does the same from any other source. Add `--akk-tenor 5Y` (or any tenor above) to pick a different tenor.
- `--akk-dump` writes `data/akk_benchmark_yields.csv` (date, then one column per tenor) and `data/akk_benchmark_yields_long.csv` (adds ISIN, bid/ask and price per row), then exits.

Eurostat's daily HU 10Y series (`irt_lt_mcby_d`), which this used before, was withdrawn in 2026-11.

## 2. Diagnose: `fit`

```bash
python rates_VectorAutoRegression/rates_VectorAutoRegression.py fit --data rates_VectorAutoRegression/data/rates_official_10y.parquet --lags bic
```

This prints ADF/KPSS tests on levels and differences, the AIC/BIC/HQ/FPE lag table, the fitted model with its stability and whiteness tests, Granger causality of each currency on HUF, the HUF equation, and the FEVD.

## 3. Optimize (multithreaded walk-forward)

```bash
python rates_VectorAutoRegression/rates_VectorAutoRegression.py optimize --data rates_VectorAutoRegression/data/rates_official_10y.parquet \
    --grid rates_VectorAutoRegression/rates_VectorAutoRegression_grid.json --horizon 20 --test-years 2 --step 5 --workers 8
```

For each parameter set, the model is refit every `--step` business days over the last `--test-years` years, and the HUF level is forecast `--horizon` business days ahead. Results are ranked by out-of-sample RMSE. `rel_rmse` is RMSE divided by the random-walk RMSE, so a value below 1 means the model beats a no-change forecast. The output also reports the direction hit rate and the coverage of the 95% band.

Grid keys:
- `lags`: integers, or `"aic"`/`"bic"`/`"hqic"` to select per refit (slower).
- `transform`: `diff` or `level`.
- `window`: rolling window in observations; 0 means expanding.
- `freq`: `D` or `W`.
- `inputs`: `ALL`, `SINGLE` (each currency alone), `SUBSETS` (all 63 combinations), or an explicit set like `"EUR+USD"`.

Use `--executor process` to run in processes instead of threads.

Writes `results/optimize_h<h>.csv` and `results/best_params.json`.

## 4. Forecast

```bash
python rates_VectorAutoRegression/rates_VectorAutoRegression.py forecast --data rates_VectorAutoRegression/data/rates_official_10y.parquet \
    --params rates_VectorAutoRegression/results/best_params.json --h 126 --alpha 0.05
```

`--h` is in business days; the default 126 is half a year. Writes `results/forecast.csv` (every currency, with `_lo` and `_hi` bands at `--alpha`) and `results/forecast_huf.png`, a fan chart with 50/80/95% bands (`--bands 0.5 0.8 0.95`). In `diff` mode, the bands cover the cumulated level path. They come from the VAR's MA representation: `sum_j C_j Σu C_j'`, with `C_j = Σ_{i≤j} Ψ_i`.

## 5. Other currencies as target

Every command takes `--target CCY` (any column of the panel) or `--target ALL`. By default, the other six currencies are the inputs. Files for a target other than HUF get a `_<CCY>` suffix.

```bash
python rates_VectorAutoRegression/rates_VectorAutoRegression.py fit      --data rates_VectorAutoRegression/data/rates_official_10y.parquet --target EUR --lags bic
python rates_VectorAutoRegression/rates_VectorAutoRegression.py optimize --data rates_VectorAutoRegression/data/rates_official_10y.parquet --grid rates_VectorAutoRegression/rates_VectorAutoRegression_grid.json \
    --target ALL --executor process --top 5     # best_params_<CCY>.json per currency
python rates_VectorAutoRegression/rates_VectorAutoRegression.py forecast --data rates_VectorAutoRegression/data/rates_official_10y.parquet --target ALL --h 126
```

Without `--params`, `forecast` uses `results/best_params[_CCY].json` for each target if that file exists, and the defaults otherwise. `--target ALL` also writes `results/forecast_all.png` (one fan chart per currency) and `results/forecast_all.csv` (last value, end-of-horizon forecast and band per currency).

`--executor process` is much faster than threads here: 9.5 s against 272 s for the 600-set HUF grid on 8 workers, because statsmodels holds the GIL.
