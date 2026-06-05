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

The configured primary source is `jugaad-data`, following `plan.txt`. Its documented `stock_df` API returns historical EOD rows, not 5-minute intraday candles, so the downloader records that limitation and then tries `openchart` as a fallback for NSE charting intraday data. NSE charting availability is not guaranteed; if both paths fail, use a historical intraday Parquet/CSV file from a broker/vendor and place it at the Drive path shown in `notebooks/01_data_download.ipynb`.

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

For real training, provide a 5-minute historical intraday file at:

```text
/content/drive/MyDrive/trading_system/data/raw/RELIANCE_NS_5m_source.parquet
```

The file can also be CSV if you update `LOCAL_INTRADAY_FILE` in notebook 01. It must include `open`, `high`, `low`, `close`, `volume`, and either a DatetimeIndex or a `datetime`/`date` column.
