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

For the Kaggle NIFTY 50 minute dataset, copy the extracted CSV files into Drive before running notebook 01:

```text
/content/drive/MyDrive/trading_system/data/raw/18/
```

Notebook 01 auto-detects `NIFTY 50_5minute.csv`, filters NSE market hours, converts all-zero index volume to synthetic volume, and writes `NIFTY_5m_intraday.parquet` plus daily context. `BANK_NIFTY_data` and `NIFTY_data` folders still work as fallback local formats.

Notebook 01 uses local data for real training. The yfinance 5-minute fallback is only for smoke-testing the pipeline and is not enough for meaningful model training.
