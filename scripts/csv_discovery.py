from __future__ import annotations

from pathlib import Path


CSV_SUFFIX_PRIORITY = {
    "_minute_new.csv": 0,
    "_minute.csv": 1,
    "_5minute.csv": 2,
}


def discover_intraday_csv_files(source_dirs: list[Path]) -> list[Path]:
    """Discover intraday CSVs and dedupe by symbol using preferred suffix order."""
    candidates: dict[str, tuple[int, int, Path]] = {}
    for source_order, source_dir in enumerate(source_dirs):
        if not source_dir.exists():
            continue
        for path in source_dir.iterdir():
            if not path.is_file() or path.name.startswith("."):
                continue
            symbol = symbol_from_intraday_csv_path(path)
            if symbol is None:
                continue
            priority = _suffix_priority(path)
            key = (priority, source_order, str(path))
            current = candidates.get(symbol)
            if current is None or key < (current[0], current[1], str(current[2])):
                candidates[symbol] = (priority, source_order, path)
    return [item[2] for symbol, item in sorted(candidates.items())]


def symbol_from_intraday_csv_path(path: Path) -> str | None:
    for suffix in CSV_SUFFIX_PRIORITY:
        if path.name.endswith(suffix):
            return path.name[: -len(suffix)]
    return None


def _suffix_priority(path: Path) -> int:
    symbol = symbol_from_intraday_csv_path(path)
    if symbol is None:
        raise ValueError(f"Unsupported intraday CSV filename: {path.name}")
    suffix = path.name[len(symbol) :]
    return CSV_SUFFIX_PRIORITY[suffix]
