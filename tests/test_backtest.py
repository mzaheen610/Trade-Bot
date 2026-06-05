from __future__ import annotations

import pandas as pd

from backtest.engine import BacktestEngine
from config import BacktestConfig, LabelConfig


def test_backtest_enters_next_candle_and_applies_sell_side_costs():
    index = pd.date_range("2026-01-01 09:15", periods=3, freq="5min")
    frame = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0],
            "high": [100.0, 101.0, 100.0],
            "low": [100.0, 99.8, 100.0],
            "close": [100.0, 100.8, 100.0],
            "signal": [1, 0, 0],
        },
        index=index,
    )
    engine = BacktestEngine(
        config=BacktestConfig(brokerage_rate=0.0003, slippage_rate=0.0, stt_sell_rate=0.00025),
        labels=LabelConfig(horizon=5, target_pct=0.005, stop_pct=0.003),
    )

    result = engine.run(frame)
    trade = result.trades.iloc[0]

    assert trade["entry_time"] == index[1]
    assert trade["exit_reason"] == "target"
    assert trade["costs"] > 0


def test_backtest_uses_stop_when_stop_and_target_same_candle():
    index = pd.date_range("2026-01-01 09:15", periods=3, freq="5min")
    frame = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0],
            "high": [100.0, 101.0, 100.0],
            "low": [100.0, 99.0, 100.0],
            "close": [100.0, 100.0, 100.0],
            "signal": [1, 0, 0],
        },
        index=index,
    )
    engine = BacktestEngine(
        config=BacktestConfig(brokerage_rate=0.0, slippage_rate=0.0, stt_sell_rate=0.0),
        labels=LabelConfig(horizon=5, target_pct=0.005, stop_pct=0.003),
    )

    result = engine.run(frame)

    assert result.trades.iloc[0]["exit_reason"] == "stop"
    assert result.trades.iloc[0]["net_pnl"] < 0


def test_backtest_forces_session_close_exit():
    index = pd.to_datetime(
        [
            "2026-01-01 15:15",
            "2026-01-01 15:20",
            "2026-01-01 15:25",
        ]
    )
    frame = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0],
            "high": [100.0, 100.1, 100.1],
            "low": [100.0, 99.9, 99.9],
            "close": [100.0, 100.0, 100.0],
            "signal": [1, 0, 0],
        },
        index=index,
    )
    engine = BacktestEngine(
        config=BacktestConfig(brokerage_rate=0.0, slippage_rate=0.0, stt_sell_rate=0.0),
        labels=LabelConfig(horizon=15, target_pct=0.05, stop_pct=0.05),
    )

    result = engine.run(frame)

    assert result.trades.iloc[0]["exit_reason"] == "session_close"

