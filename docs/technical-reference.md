# TradingBot26 Technical Reference

This document describes the current implementation of TradingBot26. It is the canonical reference for the data pipeline, feature engineering, labels, model architectures, training paths, artifacts, signal generation, backtesting, and operational caveats.

## System Overview

TradingBot26 is organized as an offline research pipeline:

```text
raw OHLCV
  -> cached raw Parquet
  -> engineered intraday and daily-context features
  -> target-before-stop labels
  -> train/validation/test split
  -> model artifacts and metadata
  -> probability columns
  -> rule-based signal filter
  -> intraday backtest reports
```

The code has three practical training modes:

- `trading-mvp` CLI: single-instrument data download, feature engineering, LightGBM training, and backtesting.
- `scripts/batch_train.py`: local batch runner that processes many CSV or cached Parquet sources one instrument at a time, including optional LSTM and GRU training.
- `notebooks/05_colab_nifty50_batch.ipynb`: Colab-oriented combined training flow that loads many processed instruments, trains one pooled LightGBM model, and trains combined LSTM/GRU sequence models.

All model targets are three-class classification outputs with the mapping `SELL=0`, `HOLD=1`, `BUY=2`.

## Data And Caching

### Sources

The data loader supports these intraday sources:

- `local-csv`: Primary source for historical training. It supports Kaggle-style files with `date,open,high,low,close,volume` and older Bank Nifty files with `Instrument,Date,Time,Open,High,Low,Close`.
- `openchart`: NSE charting-data fallback for short or provider-backed intraday pulls.
- `jugaad`: Configured default provider. The documented `stock_df` path is EOD-only, so the loader treats it as unavailable for 5-minute intraday data unless an intraday function exists in the installed package.
- `yfinance-5m`: Explicit short-history smoke-test fallback. It is not intended for meaningful historical model training.
- `cached-parquet`: Internal batch mode for already cached raw Parquet sources.

Daily context comes from either `yfinance` or `intraday-resample`. The local CSV and Colab historical-data paths typically use `intraday-resample`.

### Raw Cache

Raw data is stored under `data/raw/` using safe ticker names:

- Intraday: `<ticker>_<interval>_intraday.parquet`, for example `NIFTY_5m_intraday.parquet`.
- Daily: `<ticker>_1d_daily.parquet`, for example `NIFTY_1d_daily.parquet`.

Local CSV inputs are normalized to an OHLCV frame, filtered to NSE market hours `09:15` through `15:30`, de-duplicated by timestamp, resampled to the configured interval, and saved as Parquet. Missing volume is filled with synthetic `1.0`; all-zero volume columns are also converted to synthetic volume.

### Processed Cache

Feature-engineered datasets are written under `data/processed/`:

- `<ticker>_<interval>_features.parquet`

Each processed frame keeps raw OHLCV columns and engineered columns, plus `label` and `label_name`. The model feature list is stored separately in `artifacts/feature_config*.json`, so inference and training can preserve exact feature order.

### Colab Layout

The Colab notebooks expect persistent storage under:

```python
BASE = "/content/drive/MyDrive/trading_system/"
```

The one-notebook flow in `notebooks/05_colab_nifty50_batch.ipynb` reads and writes:

- Raw CSVs under `BASE/data/raw/...`
- Raw Parquet caches under `BASE/data/raw/`
- Processed Parquets under `BASE/data/processed/`
- Model checkpoints and metadata under `BASE/models/`
- Reports under `BASE/reports/`

The training notebook can optionally stage processed files to local `/content` storage to reduce repeated reads through the Google Drive FUSE mount.

## Feature Engineering

Feature engineering is implemented by `FeatureEngineeringPipeline` and `FeatureBuilder`.

### Intraday Features

The pipeline starts from sorted intraday OHLCV bars and builds:

- Price and candle features: `return_1`, `range_pct`, `body_pct`
- Trend features: `ema_9`, `ema_21`, `ema_200`
- Momentum: `rsi_14`, MACD line/signal/histogram
- Volatility and bands: Bollinger middle/upper/lower/width/percent-b, `atr_14`
- Intraday benchmark: session-anchored `vwap`
- Volume features: shifted `volume_roll20`, `relative_volume_20`, `obv`
- Regime flags: `ema9_above_ema21`, `close_above_ema200`, `close_above_vwap`

Model candidate columns are all numeric columns except `label` and `label_name`, so the candidate set includes normalized OHLCV, normalized indicator columns, and normalized regime flags.

### Daily Context

When `include_daily_context=True`, daily data is transformed into:

- `daily_return_1`
- `daily_ema_21`
- `daily_ema_200`
- `daily_adx_14`
- `daily_close_above_ema200`

Daily context is shifted by one daily row before merging into intraday bars. This prevents the current day from seeing same-day daily close information. The shifted daily frame is merged by normalized session date using backward as-of semantics.

### Normalization

All numeric feature candidates are transformed by trailing rolling z-score:

```text
z = (current_value - trailing_mean) / trailing_std
```

Defaults:

- Window: `200`
- Minimum periods: `200`
- Shift: `1`, so the current row is not included in its own normalization window
- Stable near-zero standard deviation windows are mapped to `0.0`

The normalized columns are named with a `z_` prefix. The raw columns remain in the processed frame for reporting, signal generation, and backtesting, but the model feature list uses normalized columns.

### Lagged Features

After normalization, the pipeline appends lagged copies of every normalized model column. Defaults:

- `lag_1`
- `lag_5`
- `lag_10`

The final model feature list is:

```text
normalized current features + normalized lagged features
```

Rows with missing values in any model feature or label are dropped before the processed Parquet is written.

### Feature Metadata

`feature_config*.json` stores:

- Ticker, symbol, interval, and source metadata
- Exact `feature_columns` order
- Label settings
- Normalization settings
- Feature settings such as lag periods and daily-context flag
- Key package versions

This file is critical because model inference must use the same feature order used during training.

## Label Construction

Labels are built by `build_forward_labels` using future high/low paths over the configured horizon.

Defaults:

- `horizon=15`
- `target_pct=0.005`
- `stop_pct=0.003`

For each row:

- A long trade target is `close * (1 + target_pct)`.
- A long stop is `close * (1 - stop_pct)`.
- A short trade target is `close * (1 - target_pct)`.
- A short stop is `close * (1 + stop_pct)`.
- The code scans the next `horizon` bars, excluding the current bar.

Resolution rules:

- `BUY`: long target is hit before long stop and before a valid short target.
- `SELL`: short target is hit before short stop and before a valid long target.
- `HOLD`: neither side reaches target, a stop is hit first, both target and stop happen in the same future bar, or long and short resolve on the same step.

The final class mapping is:

```text
SELL = 0
HOLD = 1
BUY  = 2
```

## Splitting

There are two split implementations:

- Single-instrument CLI path: `chronological_split` sorts rows by timestamp and applies row-count ratios.
- Pooled multi-instrument path: `date_based_split` sorts by timestamp, finds unique calendar dates, and splits by date boundaries so all symbols share the same train/validation/test calendar windows.

Default ratios:

```text
train = 0.70
validation = 0.15
test = 0.15
```

Sequence models build training windows only inside each split. Pooled sequence datasets group by `symbol`, so no sequence crosses from one instrument into another.

## Model Architectures

### Single-Instrument LightGBM

The CLI `trading-mvp train` path trains one LightGBM model for the configured ticker.

Core parameters:

- Objective: `multiclass`
- Classes: `3`
- Estimators: `300`
- Learning rate: `0.03`
- Leaves: `31`
- Subsample: `0.85`
- Column sample: `0.85`
- Regularization: `reg_lambda=1.0`
- Class weighting: `balanced`
- Device: CPU by default, with optional LightGBM GPU/CUDA parameters exposed in the model code

Outputs:

- Model: `artifacts/models/lgbm_<ticker>_<interval>.joblib`
- Metadata: `artifacts/models/lgbm_<ticker>_<interval>_metadata.json`
- Metrics: `reports/<ticker>_<interval>_model_metrics.json`

### Pooled LightGBM

The pooled LightGBM path loads many `*_features.parquet` files, keeps only shared numeric feature columns, combines all instruments into one frame, and uses a date-based split. It can include `symbol` as a categorical feature.

For full-universe runs, the pooled loader scans Parquet metadata first, computes the shared numeric feature intersection, then reads only those selected columns plus `label` and `symbol`. Feature columns are downcast to `float32` in memory before combining to keep RAM usage manageable.

Core parameters are similar to the single-instrument path, with:

- Estimators: `400`
- Optional categorical `symbol`
- Optional `device_type` of `cpu`, `gpu`, or `cuda`, subject to the installed LightGBM build

Outputs include global validation/test metrics and `per_instrument_test` metrics for each symbol in the test split.

### LSTM Classifier

The LSTM model consumes tensors shaped:

```text
batch x lookback x input_size
```

Default Colab/batch lookback:

```text
lookback = 60
```

Architecture:

- LSTM layer 1: `input_size -> 128`, `batch_first=True`
- Dropout: `0.3`
- LSTM layer 2: `128 -> 64`, `batch_first=True`
- Dropout: `0.3`
- Linear head: `64 -> 3`

The classifier uses the final hidden state from the second LSTM layer.

### GRU Classifier

The GRU model consumes the same sequence shape:

```text
batch x lookback x input_size
```

Architecture:

- GRU layer: `input_size -> 128`, `batch_first=True`
- Dropout: `0.3`
- Linear head: `128 -> 3`

The classifier uses the final hidden state from the GRU.

## Torch Training Architecture

Torch training is implemented in `models/torch_training.py`.

### Dataset Forms

There are two dataset paths:

- `NumpySequenceDataset`: consumes prebuilt `SequenceDatasetArrays`.
- `FrameSequenceDataset`: lazily slices per-symbol sequence windows from a dataframe grouped by `symbol`.

`FrameSequenceDataset` keeps separate feature blocks per symbol and labels each sequence by the label at the sequence end. It also reports:

- Non-finite feature values replaced
- Feature values clipped beyond `abs(value) > 20`

Torch feature cleaning converts non-finite values to `0.0` and clips features to `[-20, 20]` by default.

### Training Loop

Defaults:

- Batch size: `512`
- Learning rate: `1e-3`
- Optimizer: `AdamW`
- Loss: class-weighted `CrossEntropyLoss`
- Max grad norm: `1.0`
- Mixed precision: optional `use_amp`
- CUDA pin memory: enabled when device is CUDA
- DataLoader workers: configured by caller

The trainer validates that inputs, loss, gradients, validation loss, and checkpoint weights remain finite.

### Checkpoints

Each model writes:

- `*_latest.pt`: full resumable checkpoint with model state, optimizer state, current epoch, validation loss, best validation loss, and history
- `*_best.pt`: model state for the lowest validation loss seen so far
- `*_final.pt`: final model state after the requested epoch range
- `*_history.json`: epoch rows containing train loss, validation loss, and validation accuracy
- `*_epoch_<n>.pt`: periodic copy every 5 epochs, including epoch 0

If `*_latest.pt` exists and contains finite history and weights, training resumes from the next epoch. For full training after a smoke test, use a distinct model scope/name or remove the smoke-test checkpoints first.

## Training Modes

### CLI Single-Instrument Flow

The `trading-mvp` commands operate on one configured instrument:

```bash
trading-mvp download
trading-mvp features
trading-mvp train
trading-mvp backtest
trading-mvp run-all
trading-mvp report-summary
```

This path trains a LightGBM model and does not train Torch models. Backtesting loads the single-instrument LightGBM model, predicts probabilities on the chronological test split, generates filtered signals, and writes reports.

### Local Batch Flow

`scripts/batch_train.py` discovers many CSV or raw Parquet sources, then runs the single-instrument pipeline for each source:

```bash
python scripts/batch_train.py --source-root data/raw/nifty50 --source-format csv
```

It can also train LSTM and GRU models per source after feature engineering:

- `--skip-torch` disables Torch models
- `--no-lstm` disables LSTM
- `--no-gru` disables GRU
- `--epochs`, `--lookback`, `--batch-size`, `--learning-rate`, and `--num-workers` configure Torch training

This script is not the pooled combined trainer; it processes discovered sources one at a time.

### Pooled LightGBM Script

`scripts/train_pooled_lightgbm.py` trains one LightGBM model from many processed feature files:

```bash
python scripts/train_pooled_lightgbm.py --processed-root data/processed --output-name nifty50_combined_5m
```

It supports `--limit`, `--min-rows`, `--no-symbol-feature`, `--lightgbm-device-type`, and `--lightgbm-gpu-device-id`.

### Colab Combined Flow

`notebooks/05_colab_nifty50_batch.ipynb` is the main combined Colab workflow. It:

- Discovers raw CSV files in Drive
- Caches intraday and daily Parquet files
- Builds per-symbol processed feature files
- Loads many processed instruments
- Selects shared feature columns
- Builds one combined date-based split
- Trains pooled LightGBM with optional categorical `symbol`
- Trains combined LSTM and GRU models from per-symbol sequence datasets
- Writes combined training metadata

Important knobs in the training cell:

- `LOOKBACK`
- `EPOCHS`
- `BATCH_SIZE`
- `LEARNING_RATE`
- `NUM_WORKERS`
- `TRAIN_LIGHTGBM`
- `TRAIN_LSTM`
- `TRAIN_GRU`
- `MAX_FILES`
- `LGBM_DEVICE_TYPE`
- `STAGE_PROCESSED_TO_LOCAL`
- `SCOPE`

`MAX_FILES=5` is a smoke-test style setting. Use `MAX_FILES=0` for the full available universe.

## Outputs And Reports

### Model Artifacts

Model artifacts are written under `artifacts/models/` locally and under the configured Drive model directory in Colab.

Common artifacts:

- `lgbm_*.joblib`
- `lgbm_*_metadata.json`
- `lstm_*_latest.pt`
- `lstm_*_best.pt`
- `lstm_*_final.pt`
- `lstm_*_history.json`
- `gru_*_latest.pt`
- `gru_*_best.pt`
- `gru_*_final.pt`
- `gru_*_history.json`
- `training_*_metadata.json`

### Model Metrics

LightGBM metrics include:

- Row count
- Accuracy
- Multiclass log loss
- Label distribution

Pooled LightGBM also writes per-instrument test metrics.

Torch history currently includes:

- Epoch
- Train loss
- Validation loss
- Validation accuracy

### Backtest Reports

Backtest reports are written under `reports/`:

- `<ticker>_<interval>_signals.csv`
- `<ticker>_<interval>_trades.csv`
- `<ticker>_<interval>_equity_curve.csv`
- `<ticker>_<interval>_metrics.json`
- `<ticker>_<interval>_monthly_metrics.csv`

When a ticker-specific prefix is used, latest-run aliases are also written:

- `signals.csv`
- `trades.csv`
- `equity_curve.csv`
- `metrics.json`
- `monthly_metrics.csv`

## Signal Generation And Backtesting

### Signal Filter

`SignalFuser` joins model probabilities onto the feature frame and creates:

- `signal`: `1` for BUY, `-1` for SELL, `0` for HOLD
- `signal_name`
- `signal_confidence`
- `volume_confirmed`
- `actionable`

Defaults:

- Confidence threshold: `0.65`
- Volume multiplier: `1.5`, except local CSV CLI runs default to `0.0` unless overridden because many historical CSV files have synthetic volume

BUY requires:

- `p_buy >= confidence_threshold`
- `p_buy >= p_sell`
- volume confirmation

SELL requires:

- `p_sell >= confidence_threshold`
- `p_sell > p_buy`
- volume confirmation

### Backtest Mechanics

The backtest engine:

- Reads the previous bar signal
- Enters on the next bar open
- Allows entries from `09:20` through `15:20`
- Prevents overnight holding by forcing session-close exits
- Exits on opposite signal, target, stop, timeout, session close, or end of data
- Applies slippage on entry and exit
- Applies brokerage on entry and exit turnover
- Applies STT on sell turnover

Defaults:

- Brokerage: `0.0003`
- Slippage: `0.0005`
- STT on sell: `0.00025`
- Bars per trading day: `75`
- Starting equity: `1.0`

Reported metrics:

- Total return
- Sharpe
- Calmar
- Max drawdown
- Win rate
- Average profit/loss ratio
- Number of trades
- Trades per day
- Monthly trade summary

## Current Caveats

- Some names still reflect the original MVP stage, including the package name and `trading-mvp` CLI entrypoint.
- LightGBM GPU/CUDA training only works when the installed LightGBM build supports the requested `device_type`. The code probes non-CPU LightGBM before full pooled training.
- Low GPU memory usage during LSTM/GRU training can be normal because the current recurrent models are small and default batches are modest.
- `*_latest.pt` checkpoints resume automatically. Do not reuse a smoke-test `SCOPE` or checkpoint set for a full universe run unless that resume behavior is intentional.
- Accuracy should be compared against the validation/test class distribution baseline. A high `HOLD` share can make raw accuracy misleading.
- Torch training uses class-weighted cross entropy, so validation accuracy may move differently from validation loss.
- Large `train_clipped` or `validation_clipped` counts indicate unstable or extreme normalized feature values. Diagnose clipped feature columns before committing to long full-universe training.
- The feature pipeline currently normalizes all numeric candidates, including binary regime flags. This is implementation behavior, not necessarily the final preferred modeling choice.
- The yfinance 5-minute path is for smoke tests only and is too short for serious intraday model training.
