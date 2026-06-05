from __future__ import annotations

import numpy as np
import pandas as pd

from config import SignalConfig


class SignalFuser:
    """Single-model MVP fuser with confidence and volume confirmation filters."""

    def __init__(self, config: SignalConfig) -> None:
        self.config = config

    def generate(self, frame: pd.DataFrame, probabilities: pd.DataFrame) -> pd.DataFrame:
        required = {"volume", "volume_roll20"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Signal generation requires columns: {sorted(missing)}")
        prob_missing = {"p_sell", "p_hold", "p_buy"}.difference(probabilities.columns)
        if prob_missing:
            raise ValueError(f"Missing probability columns: {sorted(prob_missing)}")

        output = frame.join(probabilities, how="left")
        volume_ok = output["volume"] >= (
            output["volume_roll20"] * self.config.volume_multiplier
        )
        buy_ok = (
            (output["p_buy"] >= self.config.confidence_threshold)
            & (output["p_buy"] >= output["p_sell"])
            & volume_ok
        )
        sell_ok = (
            (output["p_sell"] >= self.config.confidence_threshold)
            & (output["p_sell"] > output["p_buy"])
            & volume_ok
        )
        output["signal"] = np.select([buy_ok, sell_ok], [1, -1], default=0).astype(int)
        output["signal_name"] = output["signal"].map({1: "BUY", -1: "SELL", 0: "HOLD"})
        output["signal_confidence"] = output[["p_buy", "p_sell"]].max(axis=1)
        output["volume_confirmed"] = volume_ok.fillna(False)
        output["actionable"] = output["signal"] != 0
        return output

