from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from config import MarketConfig, PathConfig, SplitConfig
from features.labels import ID_TO_LABEL, LABEL_TO_ID
from models.splits import SplitFrames, chronological_split


@dataclass(frozen=True)
class TrainingResult:
    model_path: Path
    metadata_path: Path
    metrics_path: Path
    split_frames: SplitFrames
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]


class LightGBMModel:
    def __init__(self, *, paths: PathConfig, market: MarketConfig) -> None:
        self.paths = paths
        self.market = market
        self.paths.ensure()

    def model_path(self) -> Path:
        return self.paths.model_artifact_dir / (
            f"lgbm_{self.market.ticker.replace('.', '_')}_{self.market.interval}.joblib"
        )

    def metadata_path(self) -> Path:
        return self.paths.model_artifact_dir / (
            f"lgbm_{self.market.ticker.replace('.', '_')}_{self.market.interval}_metadata.json"
        )

    def metrics_path(self) -> Path:
        return self.paths.report_dir / (
            f"{_safe_name(self.market.ticker)}_{self.market.interval}_model_metrics.json"
        )

    def train(
        self,
        df: pd.DataFrame,
        *,
        feature_columns: list[str],
        split_config: SplitConfig,
        lightgbm_device_type: str = "cpu",
        lightgbm_gpu_device_id: int = 0,
    ) -> TrainingResult:
        try:
            from lightgbm import LGBMClassifier
            from sklearn.metrics import accuracy_score, log_loss
        except ImportError as exc:
            raise RuntimeError(
                "LightGBM training dependencies are not installed. Run "
                "`pip install -e .` in the Python 3.11 environment."
            ) from exc

        splits = chronological_split(df, split_config)
        y_train = splits.train["label"].astype(int)
        present_labels = set(y_train.unique())
        if len(present_labels) < 2:
            raise ValueError(
                "Training data contains fewer than two label classes. "
                "Use a larger history window or adjust label thresholds."
            )

        lgbm_params: dict[str, Any] = {
            "objective": "multiclass",
            "num_class": 3,
            "n_estimators": 300,
            "learning_rate": 0.03,
            "max_depth": -1,
            "num_leaves": 31,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_lambda": 1.0,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 1 if lightgbm_device_type != "cpu" else -1,
            "device_type": lightgbm_device_type,
        }
        if lightgbm_device_type != "cpu":
            lgbm_params["gpu_device_id"] = lightgbm_gpu_device_id
            lgbm_params["max_bin"] = 63
        model = LGBMClassifier(**lgbm_params)
        model.fit(
            splits.train[feature_columns],
            y_train,
            eval_set=[(splits.validation[feature_columns], splits.validation["label"].astype(int))],
            eval_metric="multi_logloss",
        )

        validation_metrics = _classification_metrics(
            model,
            splits.validation,
            feature_columns,
            accuracy_score=accuracy_score,
            log_loss=log_loss,
        )
        test_metrics = _classification_metrics(
            model,
            splits.test,
            feature_columns,
            accuracy_score=accuracy_score,
            log_loss=log_loss,
        )

        model_path = self.model_path()
        metadata_path = self.metadata_path()
        metrics_path = self.metrics_path()
        joblib.dump(model, model_path)
        metadata = {
            "ticker": self.market.ticker,
            "interval": self.market.interval,
            "feature_columns": feature_columns,
            "label_mapping": LABEL_TO_ID,
            "classes_": [int(value) for value in model.classes_],
            "lightgbm_params": lgbm_params,
            "train_rows": len(splits.train),
            "validation_rows": len(splits.validation),
            "test_rows": len(splits.test),
            "train_start": str(splits.train.index.min()),
            "train_end": str(splits.train.index.max()),
            "validation_start": str(splits.validation.index.min()),
            "validation_end": str(splits.validation.index.max()),
            "test_start": str(splits.test.index.min()),
            "test_end": str(splits.test.index.max()),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        metrics_payload = {
            "ticker": self.market.ticker,
            "interval": self.market.interval,
            "validation": validation_metrics,
            "test": test_metrics,
        }
        metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True), encoding="utf-8")
        (self.paths.report_dir / "model_metrics.json").write_text(
            json.dumps(metrics_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        return TrainingResult(
            model_path=model_path,
            metadata_path=metadata_path,
            metrics_path=metrics_path,
            split_frames=splits,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
        )

    def load(self) -> Any:
        path = self.model_path()
        if not path.exists():
            raise FileNotFoundError(f"Missing LightGBM artifact: {path}")
        return joblib.load(path)

    def predict_probabilities(
        self,
        model: Any,
        df: pd.DataFrame,
        feature_columns: list[str],
    ) -> pd.DataFrame:
        probabilities = model.predict_proba(df[feature_columns])
        output = pd.DataFrame(index=df.index)
        for class_id, column_values in zip(model.classes_, probabilities.T):
            label_name = ID_TO_LABEL[int(class_id)].lower()
            output[f"p_{label_name}"] = column_values
        for label_name in ("sell", "hold", "buy"):
            column = f"p_{label_name}"
            if column not in output.columns:
                output[column] = 0.0
        return output[["p_sell", "p_hold", "p_buy"]]


def _classification_metrics(
    model: Any,
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    accuracy_score: Any,
    log_loss: Any,
) -> dict[str, Any]:
    y_true = frame["label"].astype(int)
    y_pred = model.predict(frame[feature_columns])
    y_prob = model.predict_proba(frame[feature_columns])
    labels = [0, 1, 2]
    return {
        "rows": int(len(frame)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, _align_probabilities(model.classes_, y_prob), labels=labels)),
        "label_distribution": {
            ID_TO_LABEL[int(label)]: int(count)
            for label, count in y_true.value_counts().sort_index().items()
        },
    }


def _align_probabilities(classes: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    aligned = np.zeros((probabilities.shape[0], 3), dtype=float)
    for source_index, class_id in enumerate(classes):
        aligned[:, int(class_id)] = probabilities[:, source_index]
    return aligned


def _safe_name(value: str) -> str:
    return value.replace(".", "_").replace("/", "_").replace(" ", "_")
