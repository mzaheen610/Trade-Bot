from __future__ import annotations

import argparse
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
    return parser


def add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--symbol", default="RELIANCE")
    parser.add_argument("--ticker", default="RELIANCE.NS")
    parser.add_argument("--series", default="EQ")
    parser.add_argument("--interval", default="5m")
    parser.add_argument(
        "--intraday-source",
        default="jugaad",
        choices=["jugaad", "yfinance-5m"],
        help="Primary source is jugaad; yfinance-5m is a short-history fallback.",
    )
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--normalization-window", type=int, default=200)
    parser.add_argument("--confidence-threshold", type=float, default=0.65)
    parser.add_argument("--volume-multiplier", type=float, default=1.5)


def build_config(args: argparse.Namespace) -> PipelineConfig:
    market = MarketConfig(
        symbol=args.symbol,
        ticker=args.ticker,
        series=args.series,
        interval=args.interval,
        intraday_source=args.intraday_source,
        lookback_days=args.lookback_days,
    )
    normalizer = NormalizerConfig(
        window=args.normalization_window,
        min_periods=args.normalization_window,
    )
    signals = SignalConfig(
        confidence_threshold=args.confidence_threshold,
        volume_multiplier=args.volume_multiplier,
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
    )
    print(f"metrics: {paths['metrics']}")
    print(f"monthly: {paths['monthly']}")
    print(f"trades: {paths['trades']} count={metrics['num_trades']}")


if __name__ == "__main__":
    raise SystemExit(main())

