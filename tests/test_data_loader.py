from __future__ import annotations

import sys
import types

import pandas as pd

from config import MarketConfig, PathConfig
from data.loader import DataUnavailableError, MarketDataLoader


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


class FallbackLoader(MarketDataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = []

    def _download_jugaad_intraday(self, *, start, end):
        self.calls.append("jugaad")
        raise DataUnavailableError("jugaad EOD only")

    def _download_openchart_intraday(self, *, start, end):
        self.calls.append("openchart")
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


def test_download_intraday_tries_jugaad_then_openchart_fallback(tmp_path):
    paths = PathConfig(
        root=tmp_path,
        raw_data_dir=tmp_path / "data" / "raw",
        processed_data_dir=tmp_path / "data" / "processed",
        artifact_dir=tmp_path / "artifacts",
        model_artifact_dir=tmp_path / "artifacts" / "models",
        report_dir=tmp_path / "reports",
    )
    loader = FallbackLoader(paths, MarketConfig())

    result = loader.download_intraday()

    assert result.source == "openchart"
    assert result.rows == 2
    assert loader.calls == ["jugaad", "openchart"]


def test_local_csv_source_resamples_index_ohlc_and_adds_volume(tmp_path):
    data_dir = tmp_path / "BANK_NIFTY_data"
    data_dir.mkdir()
    (data_dir / "BNF_2012.csv").write_text(
        "\n".join(
            [
                "Instrument,Date,Time,Open,High,Low,Close",
                "BANKNIFTY,20120102,09:16,100,101,99,100.5",
                "BANKNIFTY,20120102,09:17,101,102,100,101.5",
                "BANKNIFTY,20120102,09:18,102,103,101,102.5",
                "BANKNIFTY,20120102,09:19,103,104,102,103.5",
                "BANKNIFTY,20120102,09:20,104,105,103,104.5",
                "BANKNIFTY,20120102,09:21,105,106,104,105.5",
            ]
        ),
        encoding="utf-8",
    )
    paths = PathConfig(
        root=tmp_path,
        raw_data_dir=tmp_path / "data" / "raw",
        processed_data_dir=tmp_path / "data" / "processed",
        artifact_dir=tmp_path / "artifacts",
        model_artifact_dir=tmp_path / "artifacts" / "models",
        report_dir=tmp_path / "reports",
    )
    market = MarketConfig(
        symbol="BANKNIFTY",
        ticker="BANKNIFTY",
        intraday_source="local-csv",
        daily_source="intraday-resample",
        local_intraday_path=data_dir,
    )
    loader = MarketDataLoader(paths, market)

    result = loader.download_intraday(source="local-csv")
    frame = pd.read_parquet(result.path)

    assert result.rows == 2
    assert frame.iloc[0]["open"] == 100
    assert frame.iloc[0]["high"] == 105
    assert frame.iloc[0]["low"] == 99
    assert frame.iloc[0]["close"] == 104.5
    assert frame.iloc[0]["volume"] == 5


def test_local_csv_prefers_yearly_files_over_combined_exports(tmp_path):
    data_dir = tmp_path / "BANK_NIFTY_data"
    data_dir.mkdir()
    (data_dir / "BNF_2012.csv").write_text(
        "\n".join(
            [
                "Instrument,Date,Time,Open,High,Low,Close",
                "BANKNIFTY,20120102,09:16,100,101,99,100.5",
                "BANKNIFTY,20120102,09:17,101,102,100,101.5",
                "BANKNIFTY,20120102,09:18,102,103,101,102.5",
                "BANKNIFTY,20120102,09:19,103,104,102,103.5",
                "BANKNIFTY,20120102,09:20,104,105,103,104.5",
            ]
        ),
        encoding="utf-8",
    )
    (data_dir / "BNF_2010_2020.csv").write_text(
        "\n".join(
            [
                "Instrument,Date,Time,Open,High,Low,Close",
                "BANKNIFTY,20120102,09:16,900,901,899,900.5",
                "BANKNIFTY,20120102,09:17,901,902,900,901.5",
                "BANKNIFTY,20120102,09:18,902,903,901,902.5",
                "BANKNIFTY,20120102,09:19,903,904,902,903.5",
                "BANKNIFTY,20120102,09:20,904,905,903,904.5",
            ]
        ),
        encoding="utf-8",
    )
    paths = PathConfig(
        root=tmp_path,
        raw_data_dir=tmp_path / "data" / "raw",
        processed_data_dir=tmp_path / "data" / "processed",
        artifact_dir=tmp_path / "artifacts",
        model_artifact_dir=tmp_path / "artifacts" / "models",
        report_dir=tmp_path / "reports",
    )
    market = MarketConfig(
        symbol="BANKNIFTY",
        ticker="BANKNIFTY",
        intraday_source="local-csv",
        local_intraday_path=data_dir,
    )

    loader = MarketDataLoader(paths, market)
    result = loader.download_intraday(source="local-csv")
    frame = pd.read_parquet(result.path)

    assert len(frame) == 1
    assert frame.iloc[0]["open"] == 100


def test_daily_context_can_be_resampled_from_local_intraday(tmp_path):
    paths = PathConfig(
        root=tmp_path,
        raw_data_dir=tmp_path / "data" / "raw",
        processed_data_dir=tmp_path / "data" / "processed",
        artifact_dir=tmp_path / "artifacts",
        model_artifact_dir=tmp_path / "artifacts" / "models",
        report_dir=tmp_path / "reports",
    )
    market = MarketConfig(
        symbol="BANKNIFTY",
        ticker="BANKNIFTY",
        intraday_source="local-csv",
        daily_source="intraday-resample",
    )
    loader = MarketDataLoader(paths, market)
    paths.ensure()
    intraday = pd.DataFrame(
        {
            "open": [100.0, 105.0],
            "high": [106.0, 108.0],
            "low": [99.0, 104.0],
            "close": [105.0, 107.0],
            "volume": [5.0, 5.0],
        },
        index=pd.to_datetime(["2012-01-02 09:20", "2012-01-02 09:25"]),
    )
    intraday.to_parquet(loader.intraday_path())

    result = loader.download_daily_context()
    daily = pd.read_parquet(result.path)

    assert result.source == "intraday-resample"
    assert len(daily) == 1
    assert daily.iloc[0]["open"] == 100
    assert daily.iloc[0]["close"] == 107


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
