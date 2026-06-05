# Colab Notebook Pack

Run these notebooks in order:

1. `01_data_download.ipynb`
2. `02_feature_engineering.ipynb`
3. `03_train_models.ipynb`
4. `04_backtest.ipynb`

All notebooks persist outputs under:

```python
BASE = "/content/drive/MyDrive/trading_system/"
```

Before running, copy or clone this repository to:

```text
/content/drive/MyDrive/trading_system/TradingBot26/
```

The training notebook checkpoints LSTM and GRU models after every epoch and resumes from `lstm_latest.pt` / `gru_latest.pt` if present.

## Data Requirement

For the current historical CSV data, copy one of these folders into the repo or Drive before running notebook 01:

```text
/content/drive/MyDrive/trading_system/TradingBot26/data/BANK_NIFTY_data/
/content/drive/MyDrive/trading_system/data/BANK_NIFTY_data/
```

`NIFTY_data` works the same way. Notebook 01 auto-detects `BANK_NIFTY_data` first, then `NIFTY_data`, parses files like `BNF_2012.csv`, resamples 1-minute OHLC rows to 5-minute candles, and creates synthetic volume because the source files do not include volume.

Notebook 01 tries `jugaad-data` first, then `openchart` only if no local file/folder is present. The yfinance 5-minute fallback is only for smoke-testing the pipeline and is not enough for meaningful model training.
