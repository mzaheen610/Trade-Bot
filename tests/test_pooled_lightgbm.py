from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import SplitConfig
from scripts.train_pooled_lightgbm import (
    LoadedInstrument,
    combine_instruments,
    date_based_split,
    shared_feature_columns,
)


def test_pooled_split_uses_one_global_calendar_boundary() -> None:
    instruments = [
        _instrument("AAA", "2024-01-01", 10),
        _instrument("BBB", "2024-01-01", 10),
    ]
    features = shared_feature_columns(instruments)
    combined = combine_instruments(instruments, feature_columns=features)

    splits = date_based_split(
        combined,
        SplitConfig(train_ratio=0.6, validation_ratio=0.2, test_ratio=0.2),
    )

    assert features == ["feature_a", "feature_b"]
    assert splits.train.index.normalize().max() == splits.train_end_date
    assert splits.validation.index.normalize().min() > splits.train_end_date
    assert splits.validation.index.normalize().max() == splits.validation_end_date
    assert splits.test.index.normalize().min() > splits.validation_end_date
    assert set(splits.train["symbol"]) == {"AAA", "BBB"}
    assert set(splits.validation["symbol"]) == {"AAA", "BBB"}
    assert set(splits.test["symbol"]) == {"AAA", "BBB"}


def _instrument(symbol: str, start: str, periods: int) -> LoadedInstrument:
    index = pd.date_range(start, periods=periods, freq="D")
    return LoadedInstrument(
        name=symbol,
        path=Path(f"{symbol}_5m_features.parquet"),
        frame=pd.DataFrame(
            {
                "feature_a": range(periods),
                "feature_b": range(periods, periods * 2),
                "symbol": symbol,
                "label": [0, 1, 2, 1, 0] * (periods // 5),
                "label_name": ["SELL", "HOLD", "BUY", "HOLD", "SELL"] * (periods // 5),
            },
            index=index,
        ),
    )
