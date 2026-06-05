from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from config import BacktestConfig


def compute_metrics(
    trades: pd.DataFrame,
    equity_curve: pd.DataFrame,
    config: BacktestConfig,
) -> dict[str, Any]:
    if equity_curve.empty:
        return {
            "total_return": 0.0,
            "sharpe": 0.0,
            "calmar": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "avg_profit_loss_ratio": 0.0,
            "num_trades": 0,
            "trades_per_day": 0.0,
        }

    equity = equity_curve["equity"].astype(float)
    returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    annualization = math.sqrt(252 * config.bars_per_trading_day)
    sharpe = 0.0
    if len(returns) > 1 and returns.std(ddof=0) > 0:
        sharpe = float((returns.mean() / returns.std(ddof=0)) * annualization)

    drawdown = equity / equity.cummax() - 1.0
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0
    total_return = float(equity.iloc[-1] / config.starting_equity - 1.0)
    trading_days = max(1, equity_curve.index.normalize().nunique())
    annual_return = (1.0 + total_return) ** (252 / trading_days) - 1.0
    calmar = float(annual_return / abs(max_drawdown)) if max_drawdown < 0 else 0.0

    if trades.empty:
        win_rate = 0.0
        avg_profit_loss_ratio = 0.0
        trades_per_day = 0.0
    else:
        wins = trades[trades["net_pnl"] > 0]
        losses = trades[trades["net_pnl"] < 0]
        win_rate = float(len(wins) / len(trades))
        avg_win = wins["net_pnl"].mean() if not wins.empty else 0.0
        avg_loss = abs(losses["net_pnl"].mean()) if not losses.empty else 0.0
        avg_profit_loss_ratio = float(avg_win / avg_loss) if avg_loss else 0.0
        trades_per_day = float(len(trades) / trading_days)

    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "calmar": calmar,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "avg_profit_loss_ratio": avg_profit_loss_ratio,
        "num_trades": int(len(trades)),
        "trades_per_day": trades_per_day,
    }


def monthly_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    columns = ["month", "trades", "win_rate", "sharpe", "return_sum", "net_pnl"]
    if trades.empty:
        return pd.DataFrame(columns=columns)

    frame = trades.copy()
    frame["exit_time"] = pd.to_datetime(frame["exit_time"])
    frame["month"] = frame["exit_time"].dt.to_period("M").astype(str)
    rows: list[dict[str, Any]] = []
    for month, group in frame.groupby("month", sort=True):
        returns = group["return_pct"].astype(float)
        sharpe = 0.0
        if len(returns) > 1 and returns.std(ddof=0) > 0:
            sharpe = float((returns.mean() / returns.std(ddof=0)) * math.sqrt(len(returns)))
        rows.append(
            {
                "month": month,
                "trades": int(len(group)),
                "win_rate": float((group["net_pnl"] > 0).mean()),
                "sharpe": sharpe,
                "return_sum": float(returns.sum()),
                "net_pnl": float(group["net_pnl"].sum()),
            }
        )
    return pd.DataFrame(rows, columns=columns)

