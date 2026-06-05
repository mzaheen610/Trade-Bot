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

