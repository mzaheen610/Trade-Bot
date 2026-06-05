from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def write_backtest_reports(
    *,
    report_dir: Path,
    signals: pd.DataFrame,
    trades: pd.DataFrame,
    equity_curve: pd.DataFrame,
    metrics: dict[str, Any],
    monthly: pd.DataFrame,
) -> dict[str, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "signals": report_dir / "signals.csv",
        "trades": report_dir / "trades.csv",
        "equity_curve": report_dir / "equity_curve.csv",
        "metrics": report_dir / "metrics.json",
        "monthly": report_dir / "monthly_metrics.csv",
    }
    signals.to_csv(paths["signals"])
    trades.to_csv(paths["trades"], index=False)
    equity_curve.to_csv(paths["equity_curve"])
    paths["metrics"].write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    monthly.to_csv(paths["monthly"], index=False)
    return paths

