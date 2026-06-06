from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from config import PathConfig, SplitConfig
from features.labels import ID_TO_LABEL, LABEL_TO_ID


EXCLUDED_FEATURE_COLUMNS = {
    "label",
    "label_name",
    "instrument",
    "symbol",
    "ticker",
    "datetime",
    "timestamp",
}


@dataclass(frozen=True)
class LoadedInstrument:
    name: str
    path: Path
    frame: pd.DataFrame


@dataclass(frozen=True)
class DateSplitFrames:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_end_date: pd.Timestamp
    validation_end_date: pd.Timestamp


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train one pooled LightGBM model from many processed feature files."
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=PathConfig().processed_data_dir,
        help="Directory containing processed *_features.parquet files.",
    )
    parser.add_argument("--pattern", default="*_5m_features.parquet")
    parser.add_argument("--output-name", default="pooled_lgbm_5m")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-rows", type=int, default=1000)
    parser.add_argument("--no-symbol-feature", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    paths = PathConfig()
    paths.ensure()
    instruments = load_instruments(
        args.processed_root,
        pattern=args.pattern,
        min_rows=args.min_rows,
        limit=args.limit,
    )
    if len(instruments) < 2:
        raise SystemExit("Need at least two usable feature files for pooled training.")

    features = shared_feature_columns(instruments)
    if not features:
        raise SystemExit("No shared numeric feature columns found across instruments.")

    if args.dry_run:
        for instrument in instruments:
            print(f"{instrument.name}: rows={len(instrument.frame):,} features={len(features)}")
        print(f"Would train one pooled model from {len(instruments)} instruments.")
        return 0

    result = train_pooled_lgbm(
        instruments,
        feature_columns=features,
        paths=paths,
        output_name=args.output_name,
        split_config=SplitConfig(),
        include_symbol_feature=not args.no_symbol_feature,
    )
    print(f"pooled_model: {result['model_path']}")
    print(f"pooled_metadata: {result['metadata_path']}")
    print(f"pooled_metrics: {result['metrics_path']}")
    print(
        "rows: "
        f"train={result['metadata']['train_rows']:,} "
        f"validation={result['metadata']['validation_rows']:,} "
        f"test={result['metadata']['test_rows']:,}"
    )
    print(f"test_accuracy: {result['metrics']['test']['accuracy']:.4f}")
    print(f"test_log_loss: {result['metrics']['test']['log_loss']:.4f}")
    return 0


def load_instruments(
    processed_root: Path,
    *,
    pattern: str,
    min_rows: int,
    limit: int | None,
) -> list[LoadedInstrument]:
    loaded: list[LoadedInstrument] = []
    for path in sorted(processed_root.glob(pattern)):
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            print(f"Skipping {path.name}: could not read parquet ({exc}).")
            continue
        if frame.empty:
            print(f"Skipping {path.name}: empty feature file.")
            continue
        if "label" not in frame.columns:
            print(f"Skipping {path.name}: missing label column.")
            continue

        frame = frame.copy()
        frame.index = pd.to_datetime(frame.index)
        frame = frame.sort_index()
        if "symbol" not in frame.columns:
            if "instrument" in frame.columns:
                frame["symbol"] = frame["instrument"].astype(str)
            else:
                frame["symbol"] = symbol_name_from_feature_path(path)
        frame = frame.dropna(subset=["label"])
        frame["label"] = frame["label"].astype(int)
        if len(frame) < min_rows:
            print(f"Skipping {path.name}: only {len(frame):,} rows after cleanup.")
            continue

        loaded.append(
            LoadedInstrument(
                name=str(frame["symbol"].iloc[0]),
                path=path,
                frame=frame,
            )
        )
        if limit is not None and len(loaded) >= limit:
            break
    return loaded


def stage_processed_feature_files(
    source_root: Path,
    staging_root: Path,
    *,
    pattern: str = "*_5m_features.parquet",
    force_refresh: bool = False,
    limit: int | None = None,
) -> Path:
    """Copy processed feature Parquets to local storage before training.

    Colab Drive is a FUSE mount and can disconnect during many repeated Parquet
    reads. Staging reads each file once, then trains from local `/content`.
    """
    source_root = Path(source_root)
    staging_root = Path(staging_root)
    source_files = sorted(
        path
        for path in source_root.glob(pattern)
        if path.is_file() and not path.name.startswith(".")
    )
    if limit is not None:
        source_files = source_files[:limit]
    if not source_files:
        raise FileNotFoundError(f"No processed feature files found under {source_root}")

    staging_root.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped = 0
    for source_path in source_files:
        destination = staging_root / source_path.name
        if destination.exists() and not force_refresh:
            skipped += 1
            continue
        try:
            shutil.copy2(source_path, destination)
        except OSError as exc:
            if getattr(exc, "errno", None) == 107 or "Transport endpoint is not connected" in str(exc):
                raise RuntimeError(
                    "Google Drive appears disconnected while staging processed feature files. "
                    "In Colab, run `from google.colab import drive; "
                    "drive.mount('/content/drive', force_remount=True)`, then rerun setup."
                ) from exc
            raise
        copied += 1

    print(
        f"Staged {copied} processed feature files to {staging_root} "
        f"({skipped} already existed)."
    )
    return staging_root


def train_pooled_lgbm(
    instruments: list[LoadedInstrument],
    *,
    feature_columns: list[str],
    paths: PathConfig,
    output_name: str,
    split_config: SplitConfig,
    include_symbol_feature: bool = True,
) -> dict[str, Any]:
    try:
        from lightgbm import LGBMClassifier
        from sklearn.metrics import accuracy_score, log_loss
    except ImportError as exc:
        raise RuntimeError("LightGBM training dependencies are not installed.") from exc

    combined = combine_instruments(instruments, feature_columns=feature_columns)
    splits = date_based_split(combined, split_config)
    categorical_features: list[str] = []
    model_feature_columns = list(feature_columns)

    train = splits.train
    validation = splits.validation
    test = splits.test

    if include_symbol_feature:
        categories = [instrument.name for instrument in instruments]
        for frame in (train, validation, test):
            frame["symbol"] = pd.Categorical(frame["symbol"], categories=categories)
        model_feature_columns.append("symbol")
        categorical_features.append("symbol")

    if len(set(train["label"].astype(int).unique())) < 2:
        raise ValueError("Pooled training data contains fewer than two label classes.")

    model = LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=400,
        learning_rate=0.03,
        max_depth=-1,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        train[model_feature_columns],
        train["label"].astype(int),
        eval_set=[(validation[model_feature_columns], validation["label"].astype(int))],
        eval_metric="multi_logloss",
        categorical_feature=categorical_features or "auto",
    )

    validation_metrics = classification_metrics(
        model,
        validation,
        model_feature_columns,
        accuracy_score=accuracy_score,
        log_loss=log_loss,
    )
    test_metrics = classification_metrics(
        model,
        test,
        model_feature_columns,
        accuracy_score=accuracy_score,
        log_loss=log_loss,
    )
    per_instrument_test = {
        str(symbol): classification_metrics(
            model,
            symbol_frame,
            model_feature_columns,
            accuracy_score=accuracy_score,
            log_loss=log_loss,
        )
        for symbol, symbol_frame in test.groupby("symbol", observed=True)
        if not symbol_frame.empty
    }

    safe_output_name = safe_name(output_name)
    model_path = paths.model_artifact_dir / f"lgbm_{safe_output_name}.joblib"
    metadata_path = paths.model_artifact_dir / f"lgbm_{safe_output_name}_metadata.json"
    metrics_path = paths.report_dir / f"{safe_output_name}_model_metrics.json"

    metadata = {
        "model_type": "pooled_lightgbm",
        "scope": output_name,
        "output_name": output_name,
        "feature_columns": model_feature_columns,
        "base_feature_columns": feature_columns,
        "categorical_features": categorical_features,
        "label_mapping": LABEL_TO_ID,
        "classes_": [int(value) for value in model.classes_],
        "symbols": [instrument.name for instrument in instruments],
        "instruments": [
            {"name": instrument.name, "source_path": str(instrument.path), "rows": len(instrument.frame)}
            for instrument in instruments
        ],
        "split_ratios": {
            "train": split_config.train_ratio,
            "validation": split_config.validation_ratio,
            "test": split_config.test_ratio,
        },
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "train_start": str(train.index.min()),
        "train_end": str(train.index.max()),
        "train_end_date": str(splits.train_end_date.date()),
        "validation_start": str(validation.index.min()),
        "validation_end": str(validation.index.max()),
        "validation_end_date": str(splits.validation_end_date.date()),
        "test_start": str(test.index.min()),
        "test_end": str(test.index.max()),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    metrics = {
        "model_type": "pooled_lightgbm",
        "output_name": output_name,
        "validation": validation_metrics,
        "test": test_metrics,
        "per_instrument_test": per_instrument_test,
    }

    joblib.dump(model, model_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "model_path": model_path,
        "metadata_path": metadata_path,
        "metrics_path": metrics_path,
        "metadata": metadata,
        "metrics": metrics,
    }


def combine_instruments(instruments: list[LoadedInstrument], *, feature_columns: list[str]) -> pd.DataFrame:
    required = feature_columns + ["label", "symbol"]
    frames = []
    for instrument in instruments:
        frame = instrument.frame.dropna(subset=required).copy()
        frame["symbol"] = instrument.name
        frames.append(frame[required])
    combined = pd.concat(frames, axis=0)
    combined.index = pd.to_datetime(combined.index)
    return combined.sort_index()


def date_based_split(df: pd.DataFrame, config: SplitConfig) -> DateSplitFrames:
    if df.empty:
        raise ValueError("Cannot split an empty dataframe")
    ordered = df.sort_index()
    dates = pd.DatetimeIndex(ordered.index.normalize().unique()).sort_values()
    if len(dates) < 3:
        raise ValueError("Need at least three calendar dates for train/validation/test split")

    train_end_pos = max(0, int(len(dates) * config.train_ratio) - 1)
    validation_end_pos = max(
        train_end_pos + 1,
        int(len(dates) * (config.train_ratio + config.validation_ratio)) - 1,
    )
    validation_end_pos = min(validation_end_pos, len(dates) - 2)
    train_end = dates[train_end_pos]
    validation_end = dates[validation_end_pos]

    normalized = ordered.index.normalize()
    train = ordered[normalized <= train_end].copy()
    validation = ordered[(normalized > train_end) & (normalized <= validation_end)].copy()
    test = ordered[normalized > validation_end].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("Date split produced an empty train/validation/test frame")
    return DateSplitFrames(
        train=train,
        validation=validation,
        test=test,
        train_end_date=train_end,
        validation_end_date=validation_end,
    )


def classification_metrics(
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
        "log_loss": float(log_loss(y_true, align_probabilities(model.classes_, y_prob), labels=labels)),
        "label_distribution": {
            ID_TO_LABEL[int(label)]: int(count)
            for label, count in y_true.value_counts().sort_index().items()
        },
    }


def shared_feature_columns(instruments: list[LoadedInstrument]) -> list[str]:
    common = set(feature_columns(instruments[0].frame))
    for instrument in instruments[1:]:
        common &= set(feature_columns(instrument.frame))
    return [column for column in feature_columns(instruments[0].frame) if column in common]


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column not in EXCLUDED_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(frame[column])
    ]


def symbol_name_from_feature_path(path: Path) -> str:
    name = path.stem
    for suffix in ("_5m_features", "_features"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def align_probabilities(classes: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    aligned = np.zeros((probabilities.shape[0], 3), dtype=float)
    for source_index, class_id in enumerate(classes):
        aligned[:, int(class_id)] = probabilities[:, source_index]
    return aligned


def safe_name(value: str) -> str:
    return value.replace(".", "_").replace("/", "_").replace(" ", "_")


if __name__ == "__main__":
    raise SystemExit(main())
