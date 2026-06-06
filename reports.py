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
    prefix: str | None = None,
) -> dict[str, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    filename_prefix = f"{_safe_name(prefix)}_" if prefix else ""
    paths = {
        "signals": report_dir / f"{filename_prefix}signals.csv",
        "trades": report_dir / f"{filename_prefix}trades.csv",
        "equity_curve": report_dir / f"{filename_prefix}equity_curve.csv",
        "metrics": report_dir / f"{filename_prefix}metrics.json",
        "monthly": report_dir / f"{filename_prefix}monthly_metrics.csv",
    }
    signals.to_csv(paths["signals"])
    trades.to_csv(paths["trades"], index=False)
    equity_curve.to_csv(paths["equity_curve"])
    paths["metrics"].write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    monthly.to_csv(paths["monthly"], index=False)
    if prefix:
        _write_latest_aliases(
            report_dir=report_dir,
            signals=signals,
            trades=trades,
            equity_curve=equity_curve,
            metrics=metrics,
            monthly=monthly,
        )
    return paths


def _write_latest_aliases(
    *,
    report_dir: Path,
    signals: pd.DataFrame,
    trades: pd.DataFrame,
    equity_curve: pd.DataFrame,
    metrics: dict[str, Any],
    monthly: pd.DataFrame,
) -> None:
    signals.to_csv(report_dir / "signals.csv")
    trades.to_csv(report_dir / "trades.csv", index=False)
    equity_curve.to_csv(report_dir / "equity_curve.csv")
    (report_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    monthly.to_csv(report_dir / "monthly_metrics.csv", index=False)


def _safe_name(value: str | None) -> str:
    if not value:
        return ""
    return value.replace(".", "_").replace("/", "_").replace(" ", "_")
