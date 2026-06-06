"""Batch-run features, training, and backtests over per-symbol Parquet caches.

Usage:
  python scripts/batch_train.py --dry-run
  python scripts/batch_train.py --pattern "*_5m_intraday.parquet"
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from config import MarketConfig, PathConfig, PipelineConfig
from trading_cli import run_backtest, run_features, run_report_summary, run_train


def find_intraday_files(raw_dir: Path, pattern: str) -> Iterable[Path]:
    return sorted(raw_dir.glob(pattern))


def parse_names_from_intraday(path: Path) -> tuple[str, str]:
    # Expected filename like 'SBIN_NS_5m_intraday.parquet' -> prefix 'SBIN_NS'
    stem = path.stem
    if stem.endswith("_5m_intraday"):
        prefix = stem.rsplit("_5m_intraday", 1)[0]
    else:
        # fallback: take everything before first underscore group
        prefix = stem.split("_", 1)[0]
    symbol = prefix.split("_")[0]
    ticker = prefix.replace("_", ".")
    return symbol, ticker


def make_config_for(symbol: str, ticker: str) -> PipelineConfig:
    market = MarketConfig(
        symbol=symbol,
        ticker=ticker,
        intraday_source="local-csv",
        daily_source="intraday-resample",
    )
    paths = PathConfig()
    paths.ensure()
    return PipelineConfig(paths=paths, market=market)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch features+train from raw Parquet caches")
    parser.add_argument("--pattern", default="*_5m_intraday.parquet")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of symbols to run (0 = no limit)")
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Only build features and train models; do not generate backtest reports.",
    )
    args = parser.parse_args(argv)

    paths = PathConfig()
    raw = paths.raw_data_dir
    files = find_intraday_files(raw, args.pattern)
    if not files:
        print(f"No intraday Parquet files found in {raw} matching {args.pattern}")
        return 1

    count = 0
    completed = 0
    failed = 0
    for p in files:
        symbol, ticker = parse_names_from_intraday(p)
        print(f"Found: {p.name} -> symbol={symbol} ticker={ticker}")
        if args.dry_run:
            count += 1
            if args.limit and count >= args.limit:
                break
            continue
        cfg = make_config_for(symbol, ticker)
        try:
            print(f"Running features for {symbol}...")
            run_features(cfg)
            print(f"Running train for {symbol}...")
            run_train(cfg)
            if not args.skip_backtest:
                print(f"Running backtest for {symbol}...")
                run_backtest(cfg)
            print(f"Completed {symbol}\n")
            completed += 1
        except SystemExit as exc:
            print(f"Error processing {symbol}: {exc}")
            failed += 1
        except Exception as exc:  # noqa: BLE001 - surface errors per symbol
            print(f"Error processing {symbol}: {exc}")
            failed += 1
        count += 1
        if args.limit and count >= args.limit:
            break
    if not args.dry_run:
        print(f"Batch complete: completed={completed} failed={failed}\n")
        run_report_summary(paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
