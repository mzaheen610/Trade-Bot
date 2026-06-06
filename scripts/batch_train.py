"""Batch-run feature engineering and model training over many sources.

Default mode discovers cached Parquet inputs under ``data/raw``.
For the NIFTY50 CSV corpus, point ``--source-root`` at ``data/raw/nifty50`` and
set ``--source-format csv`` to get LightGBM, LSTM, and GRU training logs.

Examples:
  python scripts/batch_train.py --dry-run
  python scripts/batch_train.py --source-root data/raw/nifty50 --source-format csv --limit 1 --skip-backtest
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from config import BacktestConfig, FeatureConfig, LabelConfig, MarketConfig, NormalizerConfig, PathConfig, PipelineConfig, SignalConfig, SplitConfig
from features.sequences import SequenceBuilder
from trading_cli import run_backtest, run_download, run_features, run_report_summary, run_train


def parse_names_from_parquet(path: Path) -> tuple[str, str]:
    stem = path.stem
    if stem.endswith("_5m_intraday"):
        prefix = stem.removesuffix("_5m_intraday")
    elif stem.endswith("_1d_daily"):
        prefix = stem.removesuffix("_1d_daily")
    else:
        prefix = stem
    symbol = prefix.split("_")[0]
    ticker = prefix.replace("_", ".")
    return symbol, ticker


def parse_names_from_csv(path: Path) -> tuple[str, str]:
    stem = path.stem
    prefix = re.sub(r"_(?:5minute|15minute|60minute|minute|5m|15m|60m|day|1d)$", "", stem)
    symbol = prefix
    ticker = _safe_name(prefix)
    return symbol, ticker


def build_source_files(source_root: Path, source_format: str, source_pattern: str | None) -> list[Path]:
    if source_pattern:
        return sorted(path for path in source_root.glob(source_pattern) if path.is_file())

    if source_format == "parquet":
        return sorted(path for path in source_root.glob("*_5m_intraday.parquet") if path.is_file())
    if source_format == "csv":
        direct = sorted(path for path in source_root.glob("*_5minute.csv") if path.is_file())
        if direct:
            return direct
        return sorted(
            path
            for path in source_root.rglob("*_5minute.csv")
            if path.is_file() and ".complete" not in path.parts
        )

    parquet_files = sorted(path for path in source_root.glob("*_5m_intraday.parquet") if path.is_file())
    if parquet_files:
        return parquet_files

    csv_files = sorted(path for path in source_root.glob("*_5minute.csv") if path.is_file())
    if csv_files:
        return csv_files

    return sorted(
        path
        for path in source_root.rglob("*_5minute.csv")
        if path.is_file() and ".complete" not in path.parts
    )


def build_config_for_source(path: Path) -> tuple[PipelineConfig, bool]:
    paths = PathConfig()
    paths.ensure()

    if path.suffix.lower() == ".parquet":
        symbol, ticker = parse_names_from_parquet(path)
        market = MarketConfig(
            symbol=symbol,
            ticker=ticker,
            intraday_source="cached-parquet",
            daily_source="intraday-resample",
        )
        return PipelineConfig(paths=paths, market=market), False

    symbol, ticker = parse_names_from_csv(path)
    market = MarketConfig(
        symbol=symbol,
        ticker=ticker,
        intraday_source="local-csv",
        daily_source="intraday-resample",
        local_intraday_path=path,
    )
    return PipelineConfig(paths=paths, market=market), True


def train_torch_models(
    *,
    config: PipelineConfig,
    epochs: int,
    lookback: int,
    batch_size: int,
    learning_rate: float,
    num_workers: int,
    train_lstm: bool,
    train_gru: bool,
) -> dict[str, object]:
    from features.pipeline import FeatureEngineeringPipeline
    from models.splits import chronological_split
    import torch
    from models.torch_models import GRUClassifier, LSTMClassifier
    from models.torch_training import result_to_dict, train_torch_classifier, write_torch_training_metadata

    pipeline = FeatureEngineeringPipeline(
        paths=config.paths,
        market=config.market,
        features=config.features,
        labels=config.labels,
        normalizer=config.normalizer,
    )
    dataset = pipeline.load()
    if dataset.frame.empty:
        raise SystemExit(f"Processed feature file is empty for {config.market.ticker}")

    splits = chronological_split(dataset.frame, config.splits)
    sequence_builder = SequenceBuilder(lookback=lookback)
    train_arrays = sequence_builder.build(splits.train, feature_columns=dataset.feature_columns)
    validation_arrays = sequence_builder.build(splits.validation, feature_columns=dataset.feature_columns)
    _ = sequence_builder.build(splits.test, feature_columns=dataset.feature_columns)

    input_size = len(dataset.feature_columns)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    effective_num_workers = num_workers if device.type == "cuda" else 0
    safe_name = _safe_name(config.market.ticker)

    results: dict[str, object] = {
        "ticker": config.market.ticker,
        "symbol": config.market.symbol,
        "interval": config.market.interval,
        "lookback": lookback,
        "input_size": input_size,
        "device": str(device),
    }

    if train_lstm:
        print(f"Training LSTM for {config.market.ticker} for {epochs} epochs...")
        lstm = LSTMClassifier(input_size=input_size, hidden_size_1=128, hidden_size_2=64, dropout=0.3)
        lstm_result = train_torch_classifier(
            model=lstm,
            model_name=f"lstm_{safe_name}_{config.market.interval}",
            train_arrays=train_arrays,
            validation_arrays=validation_arrays,
            model_dir=config.paths.model_artifact_dir,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_workers=effective_num_workers,
            device=device,
        )
        results["lstm"] = result_to_dict(lstm_result)

    if train_gru:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"Training GRU for {config.market.ticker} for {epochs} epochs...")
        gru = GRUClassifier(input_size=input_size, hidden_size=128, dropout=0.3)
        gru_result = train_torch_classifier(
            model=gru,
            model_name=f"gru_{safe_name}_{config.market.interval}",
            train_arrays=train_arrays,
            validation_arrays=validation_arrays,
            model_dir=config.paths.model_artifact_dir,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_workers=effective_num_workers,
            device=device,
        )
        results["gru"] = result_to_dict(gru_result)

    metadata_path = config.paths.model_artifact_dir / f"torch_training_{safe_name}_{config.market.interval}_metadata.json"
    write_torch_training_metadata(
        metadata_path,
        {
            **results,
            "feature_config": str(pipeline.feature_config_path()),
            "feature_columns": dataset.feature_columns,
        },
    )
    print(f"torch_metadata: {metadata_path}")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch features+train from raw CSV or cached Parquet sources")
    parser.add_argument("--source-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--source-format", choices=["auto", "parquet", "csv"], default="auto")
    parser.add_argument("--pattern", default=None, help="Optional glob pattern override for the source files.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of sources to run (0 = no limit)")
    parser.add_argument("--skip-backtest", action="store_true", help="Do not run the backtest stage.")
    parser.add_argument("--skip-torch", action="store_true", help="Do not train LSTM/GRU models.")
    parser.add_argument("--no-lstm", action="store_true", help="Skip LSTM training.")
    parser.add_argument("--no-gru", action="store_true", help="Skip GRU training.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args(argv)

    paths = PathConfig()
    files = build_source_files(args.source_root, args.source_format, args.pattern)
    if not files:
        print(f"No source files found in {args.source_root}")
        return 1

    count = 0
    completed = 0
    failed = 0
    for source_path in files:
        config, needs_download = build_config_for_source(source_path)
        print(f"Found: {source_path.name} -> symbol={config.market.symbol} ticker={config.market.ticker}")
        if args.dry_run:
            count += 1
            if args.limit and count >= args.limit:
                break
            continue

        try:
            if needs_download:
                print(f"Running download for {config.market.symbol}...")
                run_download(config, force_refresh=False)
            print(f"Running features for {config.market.symbol}...")
            run_features(config)
            print(f"Running train for {config.market.symbol}...")
            run_train(config)
            if not args.skip_torch:
                torch_results = train_torch_models(
                    config=config,
                    epochs=args.epochs,
                    lookback=args.lookback,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    num_workers=args.num_workers,
                    train_lstm=not args.no_lstm,
                    train_gru=not args.no_gru,
                )
                print(f"torch_results: {torch_results}")
            if not args.skip_backtest:
                print(f"Running backtest for {config.market.symbol}...")
                run_backtest(config)
            print(f"Completed {config.market.symbol}\n")
            completed += 1
        except SystemExit as exc:
            print(f"Error processing {config.market.symbol}: {exc}")
            failed += 1
        except Exception as exc:  # noqa: BLE001 - surface errors per source
            print(f"Error processing {config.market.symbol}: {exc}")
            failed += 1

        count += 1
        if args.limit and count >= args.limit:
            break

    if not args.dry_run:
        print(f"Batch complete: completed={completed} failed={failed}\n")
        run_report_summary(paths)
    return 0


def _safe_name(value: str) -> str:
    return value.replace(".", "_").replace("/", "_").replace(" ", "_")


if __name__ == "__main__":
    raise SystemExit(main())