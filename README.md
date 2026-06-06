# TradingBot26 LightGBM MVP

This repository implements the first research slice from `plan.txt`: one NSE stock, 5-minute candles, LightGBM only, Parquet storage, trailing normalization, realistic intraday costs, and monthly backtest diagnostics.

The project is research-only. It does not place live orders.

## Setup

Use Python 3.11:

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Commands

```bash
trading-mvp download
trading-mvp features
trading-mvp train
trading-mvp backtest
trading-mvp run-all
trading-mvp report-summary
```

By default, downloads are cached and existing Parquet files are not overwritten. Pass `--force-refresh` to refresh raw data.

For the historical CSV data currently in this workspace, use the local CSV source:

```bash
trading-mvp download --intraday-source local-csv --symbol NIFTY --ticker NIFTY --force-refresh
trading-mvp features --symbol NIFTY --ticker NIFTY --intraday-source local-csv
trading-mvp train --symbol NIFTY --ticker NIFTY --intraday-source local-csv
trading-mvp backtest --symbol NIFTY --ticker NIFTY --intraday-source local-csv
```

For the older Bank Nifty folder format:

```bash
trading-mvp download --intraday-source local-csv --symbol BANKNIFTY --ticker BANKNIFTY --local-data-path data/BANK_NIFTY_data --force-refresh
trading-mvp features --symbol BANKNIFTY --ticker BANKNIFTY --intraday-source local-csv
trading-mvp train --symbol BANKNIFTY --ticker BANKNIFTY --intraday-source local-csv
trading-mvp backtest --symbol BANKNIFTY --ticker BANKNIFTY --intraday-source local-csv
```

The local CSV parser supports Kaggle files like `data/raw/18/NIFTY 50_5minute.csv` with `date,open,high,low,close,volume`, plus older files like `data/BANK_NIFTY_data/BNF_2012.csv` with `Instrument,Date,Time,Open,High,Low,Close`. It filters NSE market hours, resamples to the configured `5m` interval, and uses synthetic volume when the source file has no volume column or has an all-zero volume column.

The configured provider primary source is `jugaad-data`, following `plan.txt`. Its documented `stock_df` API returns historical EOD rows, not 5-minute intraday candles, so the downloader records that limitation and then tries `openchart` as a fallback for NSE charting intraday data. NSE charting availability is not guaranteed; for model training, prefer the local CSV path above or a broker/vendor intraday file.

The yfinance 5-minute path is available only as an explicit short-history smoke-test fallback:

```bash
trading-mvp download
trading-mvp download --intraday-source yfinance-5m
```

`feature_config.json` is written under `artifacts/` and stores feature order, label settings, normalization settings, data source metadata, and package versions.

Training and backtest outputs are written under `reports/`. Ticker-specific files are kept so runs do not overwrite each other, for example `reports/NIFTY_5m_model_metrics.json`, `reports/NIFTY_5m_metrics.json`, `reports/NIFTY_5m_trades.csv`, and `reports/NIFTY_5m_monthly_metrics.csv`. The unprefixed files such as `reports/model_metrics.json` and `reports/trades.csv` are only the latest-run aliases.

To train against the full raw NIFTY50 CSV corpus and see the same per-epoch LSTM/GRU logs you get in Colab, run:

```bash
python scripts/batch_train.py --source-root data/raw/nifty50 --source-format csv
```

For a quick smoke test on just one source, add `--limit 1` and optionally `--skip-backtest`:

```bash
python scripts/batch_train.py --source-root data/raw/nifty50 --source-format csv --limit 1 --skip-backtest
```

## Colab Training

Colab notebooks live in `notebooks/` and are designed to persist every dataset, checkpoint, model, and report under:

```python
BASE = "/content/drive/MyDrive/trading_system/"
```

Copy or clone this repository to `/content/drive/MyDrive/trading_system/TradingBot26/` before running the notebooks. The training notebook checkpoints LSTM and GRU models after every epoch and can resume from `*_latest.pt`.

For a full NIFTY50 batch run in Colab, keep the raw CSV folder under `/content/drive/MyDrive/trading_system/data/raw/nifty50/` and use the batch cells added to notebooks 01, 02, and 03. The feature-engineering step now writes per-symbol feature configs so multiple files can be processed without overwriting each other.

For a simpler one-notebook Colab flow, open [05_colab_nifty50_batch.ipynb](/Users/mzaheen/TradingBot26/notebooks/05_colab_nifty50_batch.ipynb) and run the cells from top to bottom. It performs raw cache creation, feature engineering, and training in one place.

For Colab training with the Kaggle NIFTY 50 minute dataset, put the extracted CSV files here:

```text
/content/drive/MyDrive/trading_system/data/raw/18/
```

Notebook 01 auto-detects `NIFTY 50_5minute.csv` in that folder and builds the cached Parquet files used by the later notebooks. It also still supports `BANK_NIFTY_data` and `NIFTY_data` folders.
