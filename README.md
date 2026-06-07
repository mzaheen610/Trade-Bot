# TradingBot26 Intraday Research Pipeline

TradingBot26 is a research-only NSE intraday modeling pipeline. It ingests 5-minute OHLCV data, caches raw and engineered Parquet files, builds target-before-stop labels, trains LightGBM/LSTM/GRU classifiers, and produces signal and backtest reports.

The project does not place live orders.

## Architecture Reference

For the current system design, data engineering details, feature set, model architectures, training modes, artifacts, and caveats, read [docs/technical-reference.md](docs/technical-reference.md).

## Setup

Use Python 3.11:

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

The installed CLI entrypoint is still named `trading-mvp` for compatibility.

## Commands

```bash
trading-mvp download
trading-mvp features
trading-mvp train
trading-mvp backtest
trading-mvp run-all
trading-mvp report-summary
```

The CLI path trains a single-instrument LightGBM model and can run the matching signal/backtest flow. Downloads are cached by default; pass `--force-refresh` to overwrite cached raw Parquet files.

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

The configured provider primary source is `jugaad-data`. Its documented `stock_df` API returns historical EOD rows, not 5-minute intraday candles, so the downloader records that limitation and then tries `openchart` as a fallback for NSE charting intraday data. NSE charting availability is not guaranteed; for model training, prefer the local CSV path above or a broker/vendor intraday file.

The yfinance 5-minute path is available only as an explicit short-history smoke-test fallback:

```bash
trading-mvp download
trading-mvp download --intraday-source yfinance-5m
```

`feature_config*.json` files are written under `artifacts/` and store feature order, label settings, normalization settings, data source metadata, and package versions.

Training and backtest outputs are written under `artifacts/models/` and `reports/`. Ticker-specific report files are kept so runs do not overwrite each other, for example `reports/NIFTY_5m_model_metrics.json`, `reports/NIFTY_5m_metrics.json`, `reports/NIFTY_5m_trades.csv`, and `reports/NIFTY_5m_monthly_metrics.csv`. The unprefixed files such as `reports/model_metrics.json` and `reports/trades.csv` are only the latest-run aliases.

## Batch Training

To train over a local raw CSV corpus one source at a time and see per-epoch LSTM/GRU logs, run:

```bash
python scripts/batch_train.py --source-root data/raw/nifty50 --source-format csv
```

For a quick smoke test on just one source, add `--limit 1` and optionally `--skip-backtest`:

```bash
python scripts/batch_train.py --source-root data/raw/nifty50 --source-format csv --limit 1 --skip-backtest
```

For pooled multi-instrument LightGBM training from existing processed feature files, use:

```bash
python scripts/train_pooled_lightgbm.py --processed-root data/processed --output-name nifty50_combined_5m
```

## Colab Training

Colab notebooks live in `notebooks/` and are designed to persist every dataset, checkpoint, model, and report under:

```python
BASE = "/content/drive/MyDrive/trading_system/"
```

Copy or clone this repository to `/content/drive/MyDrive/trading_system/TradingBot26/` before running the notebooks. The training notebook checkpoints LSTM and GRU models after every epoch and can resume from `*_latest.pt`.

For a full NIFTY50 batch run in Colab, keep the raw CSV folder under `/content/drive/MyDrive/trading_system/data/raw/nifty50/` and use the batch cells added to notebooks 01, 02, and 03. The feature-engineering step writes per-symbol feature configs so multiple files can be processed without overwriting each other.

For a simpler one-notebook Colab flow, open [notebooks/05_colab_nifty50_batch.ipynb](notebooks/05_colab_nifty50_batch.ipynb) and run the cells from top to bottom. It performs raw cache creation, feature engineering, pooled LightGBM training, and combined LSTM/GRU training in one place.

For Colab training with the Kaggle NIFTY 50 minute dataset, put the extracted CSV files here:

```text
/content/drive/MyDrive/trading_system/data/raw/18/
```

Notebook 01 auto-detects `NIFTY 50_5minute.csv` in that folder and builds the cached Parquet files used by the later notebooks. It also still supports `BANK_NIFTY_data` and `NIFTY_data` folders.
