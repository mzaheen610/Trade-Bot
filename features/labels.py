from __future__ import annotations

import numpy as np
import pandas as pd

from config import LabelConfig


LABEL_TO_ID = {"SELL": 0, "HOLD": 1, "BUY": 2}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


def build_forward_labels(df: pd.DataFrame, config: LabelConfig) -> pd.DataFrame:
    """Build target-before-stop labels without using them as model features."""
    required = {"high", "low", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Label construction requires columns: {sorted(missing)}")

    labels = np.full(len(df), np.nan)
    label_names: list[str | None] = [None] * len(df)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)

    for idx in range(0, len(df) - config.horizon):
        entry = closes[idx]
        if not np.isfinite(entry) or entry <= 0:
            continue

        long_target = entry * (1.0 + config.target_pct)
        long_stop = entry * (1.0 - config.stop_pct)
        short_target = entry * (1.0 - config.target_pct)
        short_stop = entry * (1.0 + config.stop_pct)

        long_step = _first_hit(
            highs[idx + 1 : idx + 1 + config.horizon],
            lows[idx + 1 : idx + 1 + config.horizon],
            target=long_target,
            stop=long_stop,
            direction=1,
        )
        short_step = _first_hit(
            highs[idx + 1 : idx + 1 + config.horizon],
            lows[idx + 1 : idx + 1 + config.horizon],
            target=short_target,
            stop=short_stop,
            direction=-1,
        )

        label_name = _resolve_label(long_step, short_step)
        labels[idx] = LABEL_TO_ID[label_name]
        label_names[idx] = label_name

    output = df.copy()
    output["label"] = labels
    output["label_name"] = label_names
    return output


def _first_hit(
    highs: np.ndarray,
    lows: np.ndarray,
    *,
    target: float,
    stop: float,
    direction: int,
) -> int | None:
    for offset, (high, low) in enumerate(zip(highs, lows), start=1):
        if direction == 1:
            target_hit = high >= target
            stop_hit = low <= stop
        else:
            target_hit = low <= target
            stop_hit = high >= stop

        if target_hit and stop_hit:
            return None
        if stop_hit:
            return None
        if target_hit:
            return offset
    return None


def _resolve_label(long_step: int | None, short_step: int | None) -> str:
    if long_step is None and short_step is None:
        return "HOLD"
    if long_step is not None and short_step is None:
        return "BUY"
    if short_step is not None and long_step is None:
        return "SELL"
    if long_step == short_step:
        return "HOLD"
    return "BUY" if long_step < short_step else "SELL"

