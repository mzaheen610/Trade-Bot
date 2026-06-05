from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config import NormalizerConfig


@dataclass(frozen=True)
class NormalizationResult:
    frame: pd.DataFrame
    normalized_columns: list[str]


class RollingZScoreNormalizer:
    def __init__(self, config: NormalizerConfig) -> None:
        self.config = config

    def transform(self, df: pd.DataFrame, columns: list[str]) -> NormalizationResult:
        output = df.copy()
        normalized_columns: list[str] = []
        values = output[columns].astype(float)
        mean = (
            values.rolling(
                window=self.config.window,
                min_periods=self.config.min_periods,
            )
            .mean()
            .shift(1)
        )
        std = (
            values.rolling(
                window=self.config.window,
                min_periods=self.config.min_periods,
            )
            .std(ddof=0)
            .shift(1)
        )
        stable_window = (std.abs() < self.config.epsilon) & mean.notna()
        zscores = (values - mean) / std.mask(stable_window)
        zscores = zscores.mask(stable_window, 0.0)

        for column in columns:
            normalized_column = f"z_{column}"
            output[normalized_column] = zscores[column]
            normalized_columns.append(normalized_column)

        return NormalizationResult(output, normalized_columns)
