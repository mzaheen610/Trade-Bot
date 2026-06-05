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

For serious training, put your historical 5-minute intraday file here before running notebook 01:

```text
/content/drive/MyDrive/trading_system/data/raw/RELIANCE_NS_5m_source.parquet
```

CSV is also supported if you update `LOCAL_INTRADAY_FILE` in notebook 01. Required fields are `open`, `high`, `low`, `close`, `volume`, plus a DatetimeIndex or `datetime`/`date` column.

Notebook 01 tries `openchart` if no file is present, but NSE charting availability can return empty data. The yfinance 5-minute fallback is only for smoke-testing the pipeline and is not enough for meaningful model training.
