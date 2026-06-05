from __future__ import annotations

import json

import pandas as pd
import pytest

from config import FeatureConfig, LabelConfig, MarketConfig, NormalizerConfig, PathConfig
from features.labels import LABEL_TO_ID, build_forward_labels
from features.normalizer import RollingZScoreNormalizer
from features.pipeline import FeatureEngineeringPipeline


def test_rolling_zscore_uses_trailing_shifted_window_only():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 100.0]})
    result = RollingZScoreNormalizer(
        NormalizerConfig(window=3, min_periods=3)
    ).transform(df, ["x"])

    expected_mean = pd.Series([1.0, 2.0, 3.0]).mean()
    expected_std = pd.Series([1.0, 2.0, 3.0]).std(ddof=0)
    assert result.frame.loc[3, "z_x"] == pytest.approx((4.0 - expected_mean) / expected_std)


def test_forward_labels_target_before_stop_and_hold_cases():
    index = pd.date_range("2026-01-01 09:15", periods=6, freq="5min")
    df = pd.DataFrame(
        {
            "open": [100, 100, 100, 100, 100, 100],
            "high": [100, 100.6, 100.1, 100.1, 100.2, 100.2],
            "low": [100, 99.8, 99.9, 99.9, 99.8, 99.8],
            "close": [100, 100, 100, 100, 100, 100],
            "volume": [1000] * 6,
        },
        index=index,
    )
    labeled = build_forward_labels(df, LabelConfig(horizon=2, target_pct=0.005, stop_pct=0.003))

    assert labeled.iloc[0]["label"] == LABEL_TO_ID["BUY"]
    assert labeled.iloc[1]["label"] == LABEL_TO_ID["HOLD"]


def test_feature_config_records_normalization_labels_and_versions(tmp_path):
    paths = PathConfig(
        root=tmp_path,
        raw_data_dir=tmp_path / "data" / "raw",
        processed_data_dir=tmp_path / "data" / "processed",
        artifact_dir=tmp_path / "artifacts",
        model_artifact_dir=tmp_path / "artifacts" / "models",
        report_dir=tmp_path / "reports",
    )
    index = pd.date_range("2026-01-01 09:15", periods=260, freq="5min")
    close = pd.Series(range(100, 360), index=index, dtype=float)
    intraday = pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1000,
        },
        index=index,
    )
    pipeline = FeatureEngineeringPipeline(
        paths=paths,
        market=MarketConfig(intraday_source="jugaad"),
        features=FeatureConfig(include_daily_context=False, lag_periods=(1,)),
        labels=LabelConfig(horizon=3),
        normalizer=NormalizerConfig(window=20, min_periods=20),
    )

    dataset = pipeline.run(intraday)
    metadata = json.loads(dataset.feature_config_path.read_text(encoding="utf-8"))

    assert dataset.frame.empty is False
    assert metadata["normalization"]["window"] == 20
    assert metadata["normalization"]["shift"] == 1
    assert metadata["labels"]["horizon"] == 3
    assert metadata["intraday_source"] == "jugaad"
    assert metadata["intraday_fallback_sources"] == ["openchart"]
    assert "jugaad-data" in metadata["package_versions"]
