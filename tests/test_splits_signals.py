from __future__ import annotations

import pandas as pd

from config import SignalConfig, SplitConfig
from models.splits import chronological_split
from strategy.signals import SignalFuser


def test_chronological_split_preserves_order_and_boundaries():
    index = pd.date_range("2026-01-01", periods=10, freq="D")
    df = pd.DataFrame({"value": range(10)}, index=index)

    splits = chronological_split(df, SplitConfig(train_ratio=0.6, validation_ratio=0.2, test_ratio=0.2))

    assert list(splits.train["value"]) == [0, 1, 2, 3, 4, 5]
    assert list(splits.validation["value"]) == [6, 7]
    assert list(splits.test["value"]) == [8, 9]
    assert splits.train.index.max() < splits.validation.index.min() < splits.test.index.min()


def test_signal_fuser_requires_confidence_and_volume_confirmation():
    index = pd.date_range("2026-01-01 09:15", periods=3, freq="5min")
    frame = pd.DataFrame(
        {
            "volume": [2000, 1000, 3000],
            "volume_roll20": [1000, 1000, 1000],
        },
        index=index,
    )
    probabilities = pd.DataFrame(
        {
            "p_sell": [0.1, 0.7, 0.2],
            "p_hold": [0.2, 0.1, 0.1],
            "p_buy": [0.7, 0.2, 0.7],
        },
        index=index,
    )

    signals = SignalFuser(SignalConfig(confidence_threshold=0.65, volume_multiplier=1.5)).generate(
        frame,
        probabilities,
    )

    assert list(signals["signal"]) == [1, 0, 1]

