from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Sequence

from backtest.engine import BacktestEngine
from backtest.metrics import compute_metrics, monthly_breakdown
from config import (
    BacktestConfig,
    FeatureConfig,
    LabelConfig,
    MarketConfig,
    NormalizerConfig,
    PathConfig,
    PipelineConfig,
    SignalConfig,
    SplitConfig,
)
from data.loader import DataUnavailableError, MarketDataLoader
from features.pipeline import FeatureEngineeringPipeline
from models.lgbm import LightGBMModel
from models.splits import chronological_split
from reports import write_backtest_reports
from strategy.signals import SignalFuser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "report-summary":
            run_report_summary(PathConfig())
            return 0
        config = build_config(args)
        if args.command == "download":
            run_download(config, force_refresh=args.force_refresh)
        elif args.command == "features":
            run_features(config)
        elif args.command == "train":
            run_train(config)
        elif args.command == "backtest":
            run_backtest(config)
        elif args.command == "run-all":
            run_download(config, force_refresh=args.force_refresh)
            run_features(config)
            run_train(config)
            run_backtest(config)
        else:
            parser.error("Missing command")
    except DataUnavailableError as exc:
        raise SystemExit(f"Data unavailable: {exc}") from exc
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LightGBM NSE intraday MVP")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("download", "features", "train", "backtest", "run-all"):
        subparser = subparsers.add_parser(command)
        add_common_options(subparser)
        if command in {"download", "run-all"}:
            subparser.add_argument(
                "--force-refresh",
                action="store_true",
                help="Overwrite cached raw Parquet files.",
            )
    subparsers.add_parser("report-summary", help="Print available per-model metrics and backtest reports.")
    return parser


def add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--symbol", default="RELIANCE")
    parser.add_argument("--ticker", default="RELIANCE.NS")
    parser.add_argument("--series", default="EQ")
    parser.add_argument("--interval", default="5m")
    parser.add_argument(
        "--intraday-source",
        default="jugaad",
        choices=["local-csv", "openchart", "jugaad", "yfinance-5m"],
        help="Primary source is jugaad; use local-csv for bundled/Kaggle historical CSV files.",
    )
    parser.add_argument(
        "--daily-source",
        choices=["yfinance", "intraday-resample"],
        default=None,
        help="Defaults to intraday-resample for local-csv, otherwise yfinance.",
    )
    parser.add_argument(
        "--local-data-path",
        type=Path,
        default=None,
        help="CSV file or folder for --intraday-source local-csv, such as data/raw/18.",
    )
    parser.add_argument(
        "--local-data-pattern",
        default="*.csv",
        help="Glob pattern used when --local-data-path points to a folder.",
    )
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--normalization-window", type=int, default=200)
    parser.add_argument("--confidence-threshold", type=float, default=0.65)
    parser.add_argument(
        "--volume-multiplier",
        type=float,
        default=None,
        help="Defaults to 0 for local-csv without real volume, otherwise 1.5.",
    )


def build_config(args: argparse.Namespace) -> PipelineConfig:
    daily_source = args.daily_source
    if daily_source is None:
        daily_source = "intraday-resample" if args.intraday_source == "local-csv" else "yfinance"
    market = MarketConfig(
        symbol=args.symbol,
        ticker=args.ticker,
        series=args.series,
        interval=args.interval,
        intraday_source=args.intraday_source,
        daily_source=daily_source,
        local_intraday_path=args.local_data_path,
        local_intraday_pattern=args.local_data_pattern,
        lookback_days=args.lookback_days,
    )
    normalizer = NormalizerConfig(
        window=args.normalization_window,
        min_periods=args.normalization_window,
    )
    signals = SignalConfig(
        confidence_threshold=args.confidence_threshold,
        volume_multiplier=(
            args.volume_multiplier
            if args.volume_multiplier is not None
            else (0.0 if args.intraday_source == "local-csv" else 1.5)
        ),
    )
    return PipelineConfig(
        paths=PathConfig(),
        market=market,
        labels=LabelConfig(),
        normalizer=normalizer,
        features=FeatureConfig(),
        splits=SplitConfig(),
        signals=signals,
        backtest=BacktestConfig(),
    )


def run_download(config: PipelineConfig, *, force_refresh: bool) -> None:
    loader = MarketDataLoader(config.paths, config.market)
    intraday = loader.download_intraday(force_refresh=force_refresh)
    daily = loader.download_daily_context(force_refresh=force_refresh)
    print(
        f"intraday: {intraday.path} rows={intraday.rows} "
        f"source={intraday.source} refreshed={intraday.refreshed}"
    )
    print(
        f"daily: {daily.path} rows={daily.rows} "
        f"source={daily.source} refreshed={daily.refreshed}"
    )


def run_features(config: PipelineConfig) -> None:
    loader = MarketDataLoader(config.paths, config.market)
    pipeline = FeatureEngineeringPipeline(
        paths=config.paths,
        market=config.market,
        features=config.features,
        labels=config.labels,
        normalizer=config.normalizer,
    )
    dataset = pipeline.run(loader.load_intraday(), loader.load_daily_context())
    print(
        f"features: {pipeline.processed_path()} rows={len(dataset.frame)} "
        f"columns={len(dataset.feature_columns)}"
    )
    print(f"feature_config: {dataset.feature_config_path}")


def run_train(config: PipelineConfig) -> None:
    pipeline = FeatureEngineeringPipeline(
        paths=config.paths,
        market=config.market,
        features=config.features,
        labels=config.labels,
        normalizer=config.normalizer,
    )
    dataset = pipeline.load()
    if dataset.frame.empty:
        raise SystemExit(
            f"Processed feature file is empty for {config.market.ticker}. "
            "Run the features command again; if it remains empty, use more intraday history "
            "or reduce the normalization window."
        )
    trainer = LightGBMModel(paths=config.paths, market=config.market)
    result = trainer.train(
        dataset.frame,
        feature_columns=dataset.feature_columns,
        split_config=config.splits,
    )
    print(f"model: {result.model_path}")
    print(f"model_metadata: {result.metadata_path}")
    print(f"model_metrics: {result.metrics_path}")


def run_backtest(config: PipelineConfig) -> None:
    pipeline = FeatureEngineeringPipeline(
        paths=config.paths,
        market=config.market,
        features=config.features,
        labels=config.labels,
        normalizer=config.normalizer,
    )
    dataset = pipeline.load()
    trainer = LightGBMModel(paths=config.paths, market=config.market)
    model = trainer.load()
    test_frame = chronological_split(dataset.frame, config.splits).test
    probabilities = trainer.predict_probabilities(model, test_frame, dataset.feature_columns)
    signals = SignalFuser(config.signals).generate(test_frame, probabilities)
    result = BacktestEngine(config=config.backtest, labels=config.labels).run(signals)
    metrics = compute_metrics(result.trades, result.equity_curve, config.backtest)
    monthly = monthly_breakdown(result.trades)
    paths = write_backtest_reports(
        report_dir=config.paths.report_dir,
        signals=signals,
        trades=result.trades,
        equity_curve=result.equity_curve,
        metrics=metrics,
        monthly=monthly,
        prefix=f"{config.market.ticker}_{config.market.interval}",
    )
    print(f"metrics: {paths['metrics']}")
    print(f"monthly: {paths['monthly']}")
    print(f"trades: {paths['trades']} count={metrics['num_trades']}")


def run_report_summary(paths: PathConfig) -> None:
    rows = []
    for metadata_path in sorted(paths.model_artifact_dir.glob("lgbm_*_metadata.json")):
        metadata = _read_json(metadata_path)
        ticker = metadata.get("ticker", metadata_path.stem)
        interval = metadata.get("interval", "5m")
        safe_ticker = ticker.replace(".", "_").replace("/", "_").replace(" ", "_")
        model_metrics = _read_json(paths.report_dir / f"{safe_ticker}_{interval}_model_metrics.json")
        backtest_metrics = _read_json(paths.report_dir / f"{safe_ticker}_{interval}_metrics.json")
        rows.append(
            {
                "ticker": ticker,
                "train_rows": metadata.get("train_rows", ""),
                "validation_rows": metadata.get("validation_rows", ""),
                "test_rows": metadata.get("test_rows", ""),
                "val_acc": _nested(model_metrics, "validation", "accuracy"),
                "test_acc": _nested(model_metrics, "test", "accuracy"),
                "test_log_loss": _nested(model_metrics, "test", "log_loss"),
                "bt_trades": backtest_metrics.get("num_trades", "") if backtest_metrics else "",
                "bt_win_rate": backtest_metrics.get("win_rate", "") if backtest_metrics else "",
                "bt_sharpe": backtest_metrics.get("sharpe", "") if backtest_metrics else "",
                "bt_return": backtest_metrics.get("total_return", "") if backtest_metrics else "",
            }
        )

    if not rows:
        print("No model metadata found under artifacts/models.")
        return

    headers = [
        "ticker",
        "train_rows",
        "validation_rows",
        "test_rows",
        "val_acc",
        "test_acc",
        "test_log_loss",
        "bt_trades",
        "bt_win_rate",
        "bt_sharpe",
        "bt_return",
    ]
    print(_format_table(rows, headers))


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _nested(data: dict, *keys: str):
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return ""
        current = current[key]
    return current


def _format_table(rows: list[dict], headers: list[str]) -> str:
    formatted_rows = [
        {header: _format_value(row.get(header, "")) for header in headers}
        for row in rows
    ]
    widths = {
        header: max(len(header), *(len(row[header]) for row in formatted_rows))
        for header in headers
    }
    lines = [
        "  ".join(header.ljust(widths[header]) for header in headers),
        "  ".join("-" * widths[header] for header in headers),
    ]
    for row in formatted_rows:
        lines.append("  ".join(row[header].ljust(widths[header]) for header in headers))
    return "\n".join(lines)


def _format_value(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
