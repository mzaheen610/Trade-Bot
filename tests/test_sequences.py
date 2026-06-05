from __future__ import annotations

import pandas as pd
import pytest

from features.sequences import SequenceBuilder


def test_sequence_builder_aligns_label_to_sequence_end():
    index = pd.date_range("2026-01-01 09:15", periods=5, freq="5min")
    frame = pd.DataFrame(
        {
            "f1": [1, 2, 3, 4, 5],
            "f2": [10, 20, 30, 40, 50],
            "label": [0, 1, 2, 1, 0],
        },
        index=index,
    )

    arrays = SequenceBuilder(lookback=3).build(frame, feature_columns=["f1", "f2"])

    assert arrays.x.shape == (3, 3, 2)
    assert arrays.y.tolist() == [2, 1, 0]
    assert arrays.index.tolist() == index[2:].tolist()
    assert arrays.x[0, :, 0].tolist() == [1, 2, 3]


def test_sequence_builder_rejects_too_few_rows():
    frame = pd.DataFrame({"f1": [1.0], "label": [0]})

    with pytest.raises(ValueError, match="Need at least"):
        SequenceBuilder(lookback=3).build(frame, feature_columns=["f1"])

