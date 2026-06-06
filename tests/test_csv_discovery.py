from __future__ import annotations

from pathlib import Path

from scripts.csv_discovery import discover_intraday_csv_files, symbol_from_intraday_csv_path


def test_discover_intraday_csv_files_accepts_supported_suffixes_and_dedupes(tmp_path: Path) -> None:
    drive_dir = tmp_path / "drive" / "nifty50stocks"
    repo_dir = tmp_path / "repo" / "nifty50stocks"
    drive_dir.mkdir(parents=True)
    repo_dir.mkdir(parents=True)
    for path in (
        drive_dir / "INFY_minute.csv",
        drive_dir / "GVTD_minute.csv",
        drive_dir / "GVTD_minute_new.csv",
        drive_dir / "NIFTY 50_5minute.csv",
        repo_dir / "INFY_minute_new.csv",
        repo_dir / "HDFCBANK_minute.csv",
        repo_dir / ".ignored_minute.csv",
        repo_dir / "README.txt",
    ):
        path.write_text("date,open,high,low,close,volume\n", encoding="utf-8")

    discovered = discover_intraday_csv_files([drive_dir, repo_dir])

    assert [path.name for path in discovered] == [
        "GVTD_minute_new.csv",
        "HDFCBANK_minute.csv",
        "INFY_minute_new.csv",
        "NIFTY 50_5minute.csv",
    ]


def test_symbol_from_intraday_csv_path_strips_supported_suffixes() -> None:
    assert symbol_from_intraday_csv_path(Path("INFY_minute.csv")) == "INFY"
    assert symbol_from_intraday_csv_path(Path("GVTD_minute_new.csv")) == "GVTD"
    assert symbol_from_intraday_csv_path(Path("NIFTY 50_5minute.csv")) == "NIFTY 50"
    assert symbol_from_intraday_csv_path(Path("NIFTY IT_day.csv")) is None
