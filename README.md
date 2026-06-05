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
```

By default, downloads are cached and existing Parquet files are not overwritten. Pass `--force-refresh` to refresh raw data.

For the historical CSV data currently in this workspace, use the local CSV source:

```bash
trading-mvp download --intraday-source local-csv --symbol BANKNIFTY --ticker BANKNIFTY --local-data-path data/BANK_NIFTY_data --force-refresh
trading-mvp features --symbol BANKNIFTY --ticker BANKNIFTY --intraday-source local-csv
trading-mvp train --symbol BANKNIFTY --ticker BANKNIFTY --intraday-source local-csv
trading-mvp backtest --symbol BANKNIFTY --ticker BANKNIFTY --intraday-source local-csv
```

The local CSV parser supports files like `data/BANK_NIFTY_data/BNF_2012.csv` with `Instrument,Date,Time,Open,High,Low,Close`. It resamples 1-minute OHLC rows to the configured `5m` interval and adds synthetic volume when the source file has no volume column.

The configured provider primary source is `jugaad-data`, following `plan.txt`. Its documented `stock_df` API returns historical EOD rows, not 5-minute intraday candles, so the downloader records that limitation and then tries `openchart` as a fallback for NSE charting intraday data. NSE charting availability is not guaranteed; for model training, prefer the local CSV path above or a broker/vendor intraday file.

The yfinance 5-minute path is available only as an explicit short-history smoke-test fallback:

```bash
trading-mvp download
trading-mvp download --intraday-source yfinance-5m
```

`feature_config.json` is written under `artifacts/` and stores feature order, label settings, normalization settings, data source metadata, and package versions.

## Colab Training

Colab notebooks live in `notebooks/` and are designed to persist every dataset, checkpoint, model, and report under:

```python
BASE = "/content/drive/MyDrive/trading_system/"
```

Copy or clone this repository to `/content/drive/MyDrive/trading_system/TradingBot26/` before running the notebooks. The training notebook checkpoints LSTM and GRU models after every epoch and can resume from `*_latest.pt`.

For Colab training with your current data, copy `BANK_NIFTY_data` or `NIFTY_data` into either the repo's `data/` folder or this Drive folder:

```text
/content/drive/MyDrive/trading_system/data/BANK_NIFTY_data/
```

Notebook 01 auto-detects those folders and builds the cached Parquet files used by the later notebooks.
