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

The primary intraday adapter is `jugaad-data`. The installed public package version may only expose historical EOD helpers such as `stock_df`; if no historical intraday helper is available, the CLI fails with a clear message instead of silently falling back to short yfinance history. The yfinance 5-minute path is available as an explicit smoke-test fallback:

```bash
trading-mvp download --intraday-source yfinance-5m
```

`feature_config.json` is written under `artifacts/` and stores feature order, label settings, normalization settings, data source metadata, and package versions.

## Colab Training

Colab notebooks live in `notebooks/` and are designed to persist every dataset, checkpoint, model, and report under:

```python
BASE = "/content/drive/MyDrive/trading_system/"
```

Copy or clone this repository to `/content/drive/MyDrive/trading_system/TradingBot26/` before running the notebooks. The training notebook checkpoints LSTM and GRU models after every epoch and can resume from `*_latest.pt`.
