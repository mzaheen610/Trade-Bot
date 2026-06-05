from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config import SplitConfig


@dataclass(frozen=True)
class SplitFrames:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def chronological_split(df: pd.DataFrame, config: SplitConfig) -> SplitFrames:
    if df.empty:
        raise ValueError("Cannot split an empty dataframe")
    ordered = df.sort_index()
    train_end = int(len(ordered) * config.train_ratio)
    validation_end = train_end + int(len(ordered) * config.validation_ratio)
    if train_end <= 0 or validation_end <= train_end or validation_end >= len(ordered):
        raise ValueError("Not enough rows for train/validation/test split")
    return SplitFrames(
        train=ordered.iloc[:train_end].copy(),
        validation=ordered.iloc[train_end:validation_end].copy(),
        test=ordered.iloc[validation_end:].copy(),
    )

