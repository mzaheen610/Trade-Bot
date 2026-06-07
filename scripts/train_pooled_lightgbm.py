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
class InstrumentSummary:
    name: str
    path: Path
    rows: int
    feature_columns: list[str]
    columns: set[str]


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
    parser.add_argument(
        "--lightgbm-device-type",
        choices=("cpu", "gpu", "cuda"),
        default="cpu",
        help="LightGBM learner device. Use cuda/gpu only with a matching LightGBM build.",
    )
    parser.add_argument("--lightgbm-gpu-device-id", type=int, default=0)
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
        lightgbm_device_type=args.lightgbm_device_type,
        lightgbm_gpu_device_id=args.lightgbm_gpu_device_id,
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
    progress_every: int = 25,
) -> list[LoadedInstrument]:
    summaries = scan_instruments(
        processed_root,
        pattern=pattern,
        min_rows=min_rows,
        limit=limit,
        progress_every=progress_every,
    )
    if not summaries:
        return []

    selected_features = shared_feature_columns_from_summaries(summaries)
    if not selected_features:
        print("No shared numeric feature columns found across usable instruments.", flush=True)
        return []

    loaded: list[LoadedInstrument] = []
    print(
        f"Loading {len(summaries):,} usable instruments with "
        f"{len(selected_features):,} shared numeric feature columns.",
        flush=True,
    )
    for file_index, summary in enumerate(summaries, start=1):
        try:
            frame = read_instrument_frame(summary, feature_columns=selected_features)
        except Exception as exc:
            print(f"Skipping {summary.path.name}: could not load selected columns ({exc}).", flush=True)
            continue

        if len(frame) < min_rows:
            print(f"Skipping {summary.path.name}: only {len(frame):,} rows after selected-column cleanup.", flush=True)
            continue

        loaded.append(
            LoadedInstrument(
                name=summary.name,
                path=summary.path,
                frame=frame,
            )
        )
        if progress_every and len(loaded) % progress_every == 0:
            print(
                f"Loaded {len(loaded):,}/{len(summaries):,} selected-column instrument frames.",
                flush=True,
            )
    print(
        f"Loaded {len(loaded):,} usable instruments using "
        f"{len(selected_features):,} shared numeric feature columns.",
        flush=True,
    )
    return loaded


def scan_instruments(
    processed_root: Path,
    *,
    pattern: str,
    min_rows: int,
    limit: int | None,
    progress_every: int = 25,
) -> list[InstrumentSummary]:
    summaries: list[InstrumentSummary] = []
    paths = sorted(processed_root.glob(pattern))
    print(f"Scanning up to {limit or len(paths)} processed feature files from {processed_root}", flush=True)
    for file_index, path in enumerate(paths, start=1):
        try:
            row_count, schema_feature_columns, columns = parquet_file_summary(path)
        except Exception as exc:
            print(f"Skipping {path.name}: could not read parquet metadata ({exc}).", flush=True)
            continue
        if row_count == 0:
            print(f"Skipping {path.name}: empty feature file.", flush=True)
            continue
        if "label" not in columns:
            print(f"Skipping {path.name}: missing label column.", flush=True)
            continue

        metadata_columns = [column for column in ("label", "symbol", "instrument") if column in columns]
        try:
            metadata = pd.read_parquet(path, columns=metadata_columns)
        except Exception as exc:
            print(f"Skipping {path.name}: could not read parquet ({exc}).", flush=True)
            continue
        metadata = metadata.dropna(subset=["label"])
        rows = len(metadata)
        if rows < min_rows:
            print(f"Skipping {path.name}: only {rows:,} rows after cleanup.", flush=True)
            continue

        if "symbol" in metadata.columns:
            name = str(metadata["symbol"].iloc[0])
        elif "instrument" in metadata.columns:
            name = str(metadata["instrument"].iloc[0])
        else:
            name = symbol_name_from_feature_path(path)

        summaries.append(
            InstrumentSummary(
                name=name,
                path=path,
                rows=rows,
                feature_columns=schema_feature_columns,
                columns=columns,
            )
        )
        if progress_every and len(summaries) % progress_every == 0:
            print(
                f"Scanned {len(summaries):,} usable instruments "
                f"after checking {file_index:,}/{len(paths):,} files.",
                flush=True,
            )
        if limit is not None and len(summaries) >= limit:
            break
    print(f"Scanned {len(summaries):,} usable instruments.", flush=True)
    return summaries


def parquet_file_summary(path: Path) -> tuple[int, list[str], set[str]]:
    try:
        import pyarrow.parquet as pq
        import pyarrow.types as pa_types
    except ImportError:
        frame = pd.read_parquet(path)
        return len(frame), feature_columns(frame), set(frame.columns)

    parquet_file = pq.ParquetFile(path)
    schema = parquet_file.schema_arrow
    names = set(schema.names)
    features = [
        field.name
        for field in schema
        if field.name not in EXCLUDED_FEATURE_COLUMNS
        and not field.name.startswith("__index_level_")
        and (
            pa_types.is_integer(field.type)
            or pa_types.is_floating(field.type)
            or pa_types.is_boolean(field.type)
        )
    ]
    return int(parquet_file.metadata.num_rows), features, names


def shared_feature_columns_from_summaries(summaries: list[InstrumentSummary]) -> list[str]:
    common = set(summaries[0].feature_columns)
    for summary in summaries[1:]:
        common &= set(summary.feature_columns)
    return [column for column in summaries[0].feature_columns if column in common]


def read_instrument_frame(summary: InstrumentSummary, *, feature_columns: list[str]) -> pd.DataFrame:
    metadata_columns = [
        column
        for column in ("label", "symbol", "instrument")
        if column in summary.columns
    ]
    read_columns = list(dict.fromkeys([*feature_columns, *metadata_columns]))
    frame = pd.read_parquet(summary.path, columns=read_columns)
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()

    if "symbol" not in frame.columns:
        if "instrument" in frame.columns:
            frame["symbol"] = frame["instrument"].astype(str)
        else:
            frame["symbol"] = summary.name
    if "instrument" in frame.columns:
        frame = frame.drop(columns=["instrument"])

    required = feature_columns + ["label"]
    frame = frame.dropna(subset=required).copy()
    frame[feature_columns] = frame[feature_columns].astype(np.float32, copy=False)
    frame["label"] = frame["label"].astype(np.int8)
    frame["symbol"] = frame["symbol"].astype("category")
    return frame[feature_columns + ["label", "symbol"]]


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
    print(
        f"Staging {len(source_files):,} processed feature files from {source_root} "
        f"to {staging_root}...",
        flush=True,
    )
    copied = 0
    skipped = 0
    for file_index, source_path in enumerate(source_files, start=1):
        destination = staging_root / source_path.name
        if destination.exists() and not force_refresh:
            if is_readable_parquet(destination):
                skipped += 1
                if file_index == 1 or file_index % 25 == 0 or file_index == len(source_files):
                    print(
                        f"Staging progress: {file_index:,}/{len(source_files):,} "
                        f"({copied:,} copied, {skipped:,} already existed)",
                        flush=True,
                    )
                continue
            print(f"Restaging unreadable local parquet: {destination.name}", flush=True)
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
        if file_index == 1 or file_index % 25 == 0 or file_index == len(source_files):
            print(
                f"Staging progress: {file_index:,}/{len(source_files):,} "
                f"({copied:,} copied, {skipped:,} already existed)",
                flush=True,
            )

    print(
        f"Staged {copied} processed feature files to {staging_root} "
        f"({skipped} already existed).",
        flush=True,
    )
    return staging_root


def is_readable_parquet(path: Path) -> bool:
    try:
        import pyarrow.parquet as pq

        pq.ParquetFile(path)
        return True
    except ImportError:
        try:
            with path.open("rb") as handle:
                if handle.seek(0, 2) < 8:
                    return False
                handle.seek(-4, 2)
                return handle.read(4) == b"PAR1"
        except OSError:
            return False
    except Exception:
        return False


def train_pooled_lgbm(
    instruments: list[LoadedInstrument],
    *,
    feature_columns: list[str],
    paths: PathConfig,
    output_name: str,
    split_config: SplitConfig,
    include_symbol_feature: bool = True,
    lightgbm_device_type: str = "cpu",
    lightgbm_gpu_device_id: int = 0,
) -> dict[str, Any]:
    print(
        f"Preparing pooled LightGBM frame from {len(instruments):,} instruments "
        f"and {len(feature_columns):,} base features...",
        flush=True,
    )
    combined = combine_instruments(instruments, feature_columns=feature_columns)
    print(f"Combined LightGBM frame rows={len(combined):,}. Splitting by calendar date...", flush=True)
    splits = date_based_split(combined, split_config)
    return train_pooled_lgbm_from_splits(
        splits,
        instruments=instruments,
        feature_columns=feature_columns,
        paths=paths,
        output_name=output_name,
        split_config=split_config,
        include_symbol_feature=include_symbol_feature,
        lightgbm_device_type=lightgbm_device_type,
        lightgbm_gpu_device_id=lightgbm_gpu_device_id,
    )


def train_pooled_lgbm_from_splits(
    splits: DateSplitFrames,
    *,
    instruments: list[LoadedInstrument],
    feature_columns: list[str],
    paths: PathConfig,
    output_name: str,
    split_config: SplitConfig,
    include_symbol_feature: bool = True,
    lightgbm_device_type: str = "cpu",
    lightgbm_gpu_device_id: int = 0,
) -> dict[str, Any]:
    try:
        from lightgbm import LGBMClassifier
        from sklearn.metrics import accuracy_score, log_loss
    except ImportError as exc:
        raise RuntimeError("LightGBM training dependencies are not installed.") from exc

    _validate_lightgbm_device(
        LGBMClassifier,
        lightgbm_device_type=lightgbm_device_type,
        lightgbm_gpu_device_id=lightgbm_gpu_device_id,
    )

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

    print(
        f"Training LightGBM: train={len(train):,}, validation={len(validation):,}, "
        f"test={len(test):,}, features={len(model_feature_columns):,}, "
        f"device_type={lightgbm_device_type}",
        flush=True,
    )
    lgbm_params: dict[str, Any] = {
        "objective": "multiclass",
        "num_class": 3,
        "n_estimators": 400,
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
    model = LGBMClassifier(
        **lgbm_params,
    )
    model.fit(
        train[model_feature_columns],
        train["label"].astype(int),
        eval_set=[(validation[model_feature_columns], validation["label"].astype(int))],
        eval_metric="multi_logloss",
        categorical_feature=categorical_features or "auto",
    )
    print("LightGBM fit complete. Computing validation/test metrics...", flush=True)

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
        "lightgbm_params": lgbm_params,
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
    print(f"Saved pooled LightGBM artifacts to {model_path}", flush=True)

    return {
        "model_path": model_path,
        "metadata_path": metadata_path,
        "metrics_path": metrics_path,
        "metadata": metadata,
        "metrics": metrics,
    }


def _validate_lightgbm_device(
    lgbm_classifier: Any,
    *,
    lightgbm_device_type: str,
    lightgbm_gpu_device_id: int,
) -> None:
    if lightgbm_device_type == "cpu":
        return

    try:
        probe = lgbm_classifier(
            objective="multiclass",
            num_class=3,
            n_estimators=1,
            min_data_in_leaf=1,
            min_data_in_bin=1,
            device_type=lightgbm_device_type,
            gpu_device_id=lightgbm_gpu_device_id,
            max_bin=63,
            verbosity=-1,
        )
        probe.fit(
            np.array(
                [
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [2.0, 0.0],
                    [2.0, 1.0],
                ],
                dtype=np.float32,
            ),
            np.array([0, 1, 2, 0, 1, 2], dtype=np.int64),
        )
    except Exception as exc:
        raise RuntimeError(
            f"LightGBM device_type={lightgbm_device_type!r} is not usable in this runtime. "
            "Install or build LightGBM with GPU/CUDA support, or set LGBM_DEVICE_TYPE='cpu'."
        ) from exc


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
