from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SequenceDatasetArrays:
    x: np.ndarray
    y: np.ndarray
    index: pd.DatetimeIndex


class SequenceBuilder:
    def __init__(self, lookback: int = 60) -> None:
        if lookback <= 1:
            raise ValueError("lookback must be greater than 1")
        self.lookback = lookback

    def build(
        self,
        frame: pd.DataFrame,
        *,
        feature_columns: list[str],
        label_column: str = "label",
    ) -> SequenceDatasetArrays:
        if len(frame) < self.lookback:
            raise ValueError(
                f"Need at least {self.lookback} rows to build sequences; got {len(frame)}"
            )
        missing = [column for column in feature_columns + [label_column] if column not in frame.columns]
        if missing:
            raise ValueError(f"Sequence dataframe missing columns: {missing}")

        ordered = frame.sort_index()
        features = ordered[feature_columns].to_numpy(dtype=np.float32)
        labels = ordered[label_column].to_numpy(dtype=np.int64)
        samples = len(ordered) - self.lookback + 1
        x = np.empty((samples, self.lookback, len(feature_columns)), dtype=np.float32)
        y = np.empty(samples, dtype=np.int64)
        index_values = []
        for sample_idx in range(samples):
            end_idx = sample_idx + self.lookback
            x[sample_idx] = features[sample_idx:end_idx]
            y[sample_idx] = labels[end_idx - 1]
            index_values.append(ordered.index[end_idx - 1])
        return SequenceDatasetArrays(x=x, y=y, index=pd.DatetimeIndex(index_values))

