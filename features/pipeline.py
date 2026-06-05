from __future__ import annotations

import importlib.metadata
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import FeatureConfig, LabelConfig, MarketConfig, NormalizerConfig, PathConfig
from features.indicators import adx, anchored_vwap, atr, bollinger_bands, ema, macd, obv, rsi
from features.labels import build_forward_labels
from features.normalizer import RollingZScoreNormalizer


@dataclass(frozen=True)
class ProcessedDataset:
    frame: pd.DataFrame
    feature_columns: list[str]
    feature_config_path: Path


class FeatureBuilder:
    def __init__(self, config: FeatureConfig) -> None:
        self.config = config

    def build(self, intraday: pd.DataFrame, daily: pd.DataFrame | None = None) -> pd.DataFrame:
        df = intraday.copy().sort_index()
        df["return_1"] = df["close"].pct_change()
        df["range_pct"] = (df["high"] - df["low"]) / df["close"].replace(0.0, np.nan)
        df["body_pct"] = (df["close"] - df["open"]) / df["open"].replace(0.0, np.nan)
        df["ema_9"] = ema(df["close"], 9)
        df["ema_21"] = ema(df["close"], 21)
        df["ema_200"] = ema(df["close"], 200)
        df["rsi_14"] = rsi(df["close"], 14)
        df = pd.concat([df, macd(df["close"])], axis=1)
        df = pd.concat([df, bollinger_bands(df["close"])], axis=1)
        df["atr_14"] = atr(df["high"], df["low"], df["close"], 14)
        df["vwap"] = anchored_vwap(df)
        df["volume_roll20"] = df["volume"].rolling(
            window=self.config.volume_confirm_window,
            min_periods=self.config.volume_confirm_window,
        ).mean().shift(1)
        df["relative_volume_20"] = df["volume"] / df["volume_roll20"].replace(0.0, np.nan)
        df["obv"] = obv(df["close"], df["volume"])
        df["ema9_above_ema21"] = (df["ema_9"] > df["ema_21"]).astype(float)
        df["close_above_ema200"] = (df["close"] > df["ema_200"]).astype(float)
        df["close_above_vwap"] = (df["close"] > df["vwap"]).astype(float)

        if self.config.include_daily_context and daily is not None and not daily.empty:
            df = self._merge_daily_context(df, daily)

        return df

    def model_feature_candidates(self, df: pd.DataFrame) -> list[str]:
        excluded = {
            "label",
            "label_name",
        }
        numeric = df.select_dtypes(include=[np.number]).columns
        return [column for column in numeric if column not in excluded]

    def _merge_daily_context(self, intraday: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
        daily_features = daily.copy().sort_index()
        daily_features["daily_return_1"] = daily_features["close"].pct_change()
        daily_features["daily_ema_21"] = ema(daily_features["close"], 21)
        daily_features["daily_ema_200"] = ema(daily_features["close"], 200)
        daily_features["daily_adx_14"] = adx(
            daily_features["high"],
            daily_features["low"],
            daily_features["close"],
            14,
        )
        daily_features["daily_close_above_ema200"] = (
            daily_features["close"] > daily_features["daily_ema_200"]
        ).astype(float)

        context_columns = [
            "daily_return_1",
            "daily_ema_21",
            "daily_ema_200",
            "daily_adx_14",
            "daily_close_above_ema200",
        ]
        shifted = daily_features[context_columns].shift(1).dropna(how="all")
        shifted = shifted.reset_index().rename(columns={"index": "session_date"})
        shifted["session_date"] = pd.to_datetime(shifted["session_date"])

        intraday_reset = intraday.reset_index().rename(columns={"index": "timestamp"})
        intraday_reset["session_date"] = pd.to_datetime(intraday_reset["timestamp"].dt.date)
        merged = pd.merge_asof(
            intraday_reset.sort_values("session_date"),
            shifted.sort_values("session_date"),
            on="session_date",
            direction="backward",
        )
        merged = merged.sort_values("timestamp").set_index("timestamp")
        merged.index.name = intraday.index.name
        return merged.drop(columns=["session_date"])


class FeatureEngineeringPipeline:
    def __init__(
        self,
        *,
        paths: PathConfig,
        market: MarketConfig,
        features: FeatureConfig,
        labels: LabelConfig,
        normalizer: NormalizerConfig,
    ) -> None:
        self.paths = paths
        self.market = market
        self.feature_config = features
        self.label_config = labels
        self.normalizer_config = normalizer
        self.paths.ensure()

    def processed_path(self) -> Path:
        return self.paths.processed_data_dir / (
            f"{self.market.ticker.replace('.', '_')}_{self.market.interval}_features.parquet"
        )

    def feature_config_path(self) -> Path:
        return self.paths.artifact_dir / "feature_config.json"

    def run(self, intraday: pd.DataFrame, daily: pd.DataFrame | None = None) -> ProcessedDataset:
        builder = FeatureBuilder(self.feature_config)
        built = builder.build(intraday, daily)
        labeled = build_forward_labels(built, self.label_config)
        candidate_columns = builder.model_feature_candidates(labeled)
        normalized = RollingZScoreNormalizer(self.normalizer_config).transform(
            labeled,
            candidate_columns,
        )
        frame = normalized.frame
        lag_columns = self._add_lagged_features(frame, normalized.normalized_columns)
        feature_columns = normalized.normalized_columns + lag_columns
        clean = frame.dropna(subset=feature_columns + ["label"]).copy()
        clean["label"] = clean["label"].astype(int)

        output_path = self.processed_path()
        clean.to_parquet(output_path)
        feature_config_path = self.feature_config_path()
        write_feature_config(
            feature_config_path,
            market=self.market,
            feature_columns=feature_columns,
            labels=self.label_config,
            normalizer=self.normalizer_config,
            features=self.feature_config,
            source_path=str(output_path),
        )
        return ProcessedDataset(clean, feature_columns, feature_config_path)

    def load(self) -> ProcessedDataset:
        output_path = self.processed_path()
        config_path = self.feature_config_path()
        if not output_path.exists():
            raise FileNotFoundError(f"Missing processed feature file: {output_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"Missing feature config file: {config_path}")
        with config_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        return ProcessedDataset(
            pd.read_parquet(output_path),
            list(metadata["feature_columns"]),
            config_path,
        )

    def _add_lagged_features(self, df: pd.DataFrame, columns: list[str]) -> list[str]:
        lag_columns: list[str] = []
        for lag in self.feature_config.lag_periods:
            for column in columns:
                lagged = f"{column}_lag_{lag}"
                df[lagged] = df[column].shift(lag)
                lag_columns.append(lagged)
        return lag_columns


def write_feature_config(
    path: Path,
    *,
    market: MarketConfig,
    feature_columns: list[str],
    labels: LabelConfig,
    normalizer: NormalizerConfig,
    features: FeatureConfig,
    source_path: str,
) -> None:
    metadata: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ticker": market.ticker,
        "symbol": market.symbol,
        "interval": market.interval,
        "intraday_source": market.intraday_source,
        "intraday_fallback_sources": list(market.intraday_fallback_sources),
        "daily_source": market.daily_source,
        "source_path": source_path,
        "feature_columns": feature_columns,
        "normalization": {
            "method": "trailing_rolling_zscore",
            "window": normalizer.window,
            "min_periods": normalizer.min_periods,
            "shift": 1,
        },
        "labels": {
            "horizon": labels.horizon,
            "target_pct": labels.target_pct,
            "stop_pct": labels.stop_pct,
        },
        "feature_settings": {
            "lag_periods": list(features.lag_periods),
            "volume_confirm_window": features.volume_confirm_window,
            "include_daily_context": features.include_daily_context,
        },
        "package_versions": _package_versions(
            [
                "openchart",
                "jugaad-data",
                "yfinance",
                "pandas",
                "numpy",
                "lightgbm",
                "scikit-learn",
            ]
        ),
    }
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def _package_versions(packages: list[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions
