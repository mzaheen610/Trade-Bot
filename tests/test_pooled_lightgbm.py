from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from config import SplitConfig
from scripts.train_pooled_lightgbm import (
    LoadedInstrument,
    combine_instruments,
    date_based_split,
    shared_feature_columns,
    stage_processed_feature_files,
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


def test_stage_processed_feature_files_copies_matching_parquets(tmp_path: Path) -> None:
    source_root = tmp_path / "drive" / "processed"
    staging_root = tmp_path / "local" / "processed"
    source_root.mkdir(parents=True)
    (source_root / "AAA_5m_features.parquet").write_text("aaa", encoding="utf-8")
    (source_root / "BBB_5m_features.parquet").write_text("bbb", encoding="utf-8")
    (source_root / "ignored.txt").write_text("ignored", encoding="utf-8")

    result = stage_processed_feature_files(source_root, staging_root)

    assert result == staging_root
    assert sorted(path.name for path in staging_root.iterdir()) == [
        "AAA_5m_features.parquet",
        "BBB_5m_features.parquet",
    ]


def test_stage_processed_feature_files_reports_drive_disconnect(tmp_path: Path) -> None:
    source_root = tmp_path / "drive" / "processed"
    staging_root = tmp_path / "local" / "processed"
    source_root.mkdir(parents=True)
    (source_root / "AAA_5m_features.parquet").write_text("aaa", encoding="utf-8")

    with patch(
        "scripts.train_pooled_lightgbm.shutil.copy2",
        side_effect=OSError(107, "Transport endpoint is not connected"),
    ):
        with pytest.raises(RuntimeError, match="Google Drive appears disconnected"):
            stage_processed_feature_files(source_root, staging_root)


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
