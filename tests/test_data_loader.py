from __future__ import annotations

import sys
import types

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


def test_openchart_reliance_uses_direct_equity_token(tmp_path, monkeypatch):
    calls = {}

    class FakeNSEData:
        def historical_direct(self, **kwargs):
            calls.update(kwargs)
            index = pd.date_range("2026-01-01 09:15", periods=2, freq="5min")
            return pd.DataFrame(
                {
                    "Open": [100, 101],
                    "High": [101, 102],
                    "Low": [99, 100],
                    "Close": [100.5, 101.5],
                    "Volume": [1000, 1100],
                },
                index=index,
            )

    fake_module = types.SimpleNamespace(NSEData=FakeNSEData)
    monkeypatch.setitem(sys.modules, "openchart", fake_module)

    paths = PathConfig(
        root=tmp_path,
        raw_data_dir=tmp_path / "data" / "raw",
        processed_data_dir=tmp_path / "data" / "processed",
        artifact_dir=tmp_path / "artifacts",
        model_artifact_dir=tmp_path / "artifacts" / "models",
        report_dir=tmp_path / "reports",
    )
    loader = MarketDataLoader(paths, MarketConfig(intraday_source="openchart"))

    result = loader.download_intraday(source="openchart")

    assert result.rows == 2
    assert calls["token"] == "2885"
    assert calls["symbol"] == "RELIANCE-EQ"
    assert calls["symbol_type"] == "Equity"
