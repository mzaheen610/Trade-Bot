from __future__ import annotations

import pandas as pd

from config import MarketConfig, PathConfig
from data.loader import MarketDataLoader


class DummyLoader(MarketDataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = 0

    def _download_yfinance_intraday(self, *, start, end):
        self.calls += 1
        index = pd.date_range("2026-01-01 09:15", periods=2, freq="5min")
        return pd.DataFrame(
            {
                "open": [100, 101],
                "high": [101, 102],
                "low": [99, 100],
                "close": [100.5, 101.5],
                "volume": [1000, 1100],
            },
            index=index,
        )


def test_download_intraday_uses_cache_without_force_refresh(tmp_path):
    paths = PathConfig(
        root=tmp_path,
        raw_data_dir=tmp_path / "data" / "raw",
        processed_data_dir=tmp_path / "data" / "processed",
        artifact_dir=tmp_path / "artifacts",
        model_artifact_dir=tmp_path / "artifacts" / "models",
        report_dir=tmp_path / "reports",
    )
    market = MarketConfig(intraday_source="yfinance-5m")
    loader = DummyLoader(paths, market)

    first = loader.download_intraday(source="yfinance-5m")
    second = loader.download_intraday(source="yfinance-5m")

    assert first.refreshed is True
    assert second.refreshed is False
    assert loader.calls == 1
    assert first.path.exists()

