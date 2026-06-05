from __future__ import annotations

import inspect
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from config import MarketConfig, PathConfig


class DataUnavailableError(RuntimeError):
    """Raised when the configured market-data source cannot produce OHLCV bars."""


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    source: str
    rows: int
    refreshed: bool


class MarketDataLoader:
    def __init__(self, paths: PathConfig, market: MarketConfig) -> None:
        self.paths = paths
        self.market = market
        self.paths.ensure()

    def intraday_path(self) -> Path:
        return self.paths.raw_data_dir / (
            f"{_safe_name(self.market.ticker)}_{self.market.interval}_intraday.parquet"
        )

    def daily_path(self) -> Path:
        return self.paths.raw_data_dir / f"{_safe_name(self.market.ticker)}_1d_daily.parquet"

    def download_intraday(
        self,
        *,
        force_refresh: bool = False,
        source: str | None = None,
        start: date | None = None,
        end: date | None = None,
    ) -> DownloadResult:
        source = source or self.market.intraday_source
        output_path = self.intraday_path()
        if output_path.exists() and not force_refresh:
            df = pd.read_parquet(output_path)
            return DownloadResult(output_path, source, len(df), refreshed=False)

        start = start or self.market.default_start
        end = end or self.market.default_end

        if source == "jugaad":
            df = self._download_jugaad_intraday(start=start, end=end)
        elif source == "yfinance-5m":
            df = self._download_yfinance_intraday(start=start, end=end)
        else:
            raise ValueError(f"Unknown intraday source: {source}")

        df = standardize_ohlcv(df, intraday=True, timezone=self.market.timezone)
        if df.empty:
            raise DataUnavailableError(f"{source} returned no intraday bars for {self.market.symbol}")

        df.to_parquet(output_path)
        return DownloadResult(output_path, source, len(df), refreshed=True)

    def download_daily_context(
        self,
        *,
        force_refresh: bool = False,
        start: date | None = None,
        end: date | None = None,
    ) -> DownloadResult:
        output_path = self.daily_path()
        if output_path.exists() and not force_refresh:
            df = pd.read_parquet(output_path)
            return DownloadResult(output_path, self.market.daily_source, len(df), refreshed=False)

        start = start or date(2000, 1, 1)
        end = end or self.market.default_end
        df = self._download_yfinance_daily(start=start, end=end)
        df = standardize_ohlcv(df, intraday=False, timezone=self.market.timezone)
        if df.empty:
            raise DataUnavailableError(f"yfinance returned no daily bars for {self.market.ticker}")

        df.to_parquet(output_path)
        return DownloadResult(output_path, self.market.daily_source, len(df), refreshed=True)

    def load_intraday(self) -> pd.DataFrame:
        path = self.intraday_path()
        if not path.exists():
            raise FileNotFoundError(f"Missing intraday cache: {path}")
        return pd.read_parquet(path)

    def load_daily_context(self) -> pd.DataFrame:
        path = self.daily_path()
        if not path.exists():
            raise FileNotFoundError(f"Missing daily cache: {path}")
        return pd.read_parquet(path)

    def _download_jugaad_intraday(self, *, start: date, end: date) -> pd.DataFrame:
        try:
            import jugaad_data.nse as nse
        except ImportError as exc:
            raise DataUnavailableError(
                "jugaad-data is not installed. Install dependencies or use "
                "--intraday-source yfinance-5m for a smoke-test fallback."
            ) from exc

        candidate_names = (
            "stock_intraday_df",
            "intraday_stock_df",
            "stock_intraday",
            "intraday_df",
        )
        for name in candidate_names:
            candidate = getattr(nse, name, None)
            if callable(candidate):
                try:
                    df = _call_jugaad_candidate(
                        candidate,
                        symbol=self.market.symbol,
                        series=self.market.series,
                        interval=self.market.interval,
                        start=start,
                        end=end,
                    )
                except TypeError:
                    continue
                if isinstance(df, pd.DataFrame) and not df.empty:
                    return df

        raise DataUnavailableError(
            "The installed jugaad-data package does not expose a historical "
            "intraday OHLCV dataframe function. Keep this source as primary "
            "when available, or run with --intraday-source yfinance-5m only for "
            "a short smoke test."
        )

    def _download_yfinance_intraday(self, *, start: date, end: date) -> pd.DataFrame:
        import yfinance as yf

        return yf.download(
            self.market.ticker,
            start=start.isoformat(),
            end=end.isoformat(),
            interval=self.market.interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )

    def _download_yfinance_daily(self, *, start: date, end: date) -> pd.DataFrame:
        import yfinance as yf

        return yf.download(
            self.market.ticker,
            start=start.isoformat(),
            end=end.isoformat(),
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )


def _call_jugaad_candidate(
    candidate: Callable[..., Any],
    *,
    symbol: str,
    series: str,
    interval: str,
    start: date,
    end: date,
) -> Any:
    signature = inspect.signature(candidate)
    kwargs: dict[str, Any] = {}
    for param in signature.parameters.values():
        name = param.name
        if name in {"symbol", "stock", "scrip"}:
            kwargs[name] = symbol
        elif name in {"series"}:
            kwargs[name] = series
        elif name in {"interval", "timeframe", "resolution"}:
            kwargs[name] = interval
        elif name in {"from_date", "start_date", "start"}:
            kwargs[name] = start
        elif name in {"to_date", "end_date", "end"}:
            kwargs[name] = end

    missing_required = [
        param.name
        for param in signature.parameters.values()
        if param.default is inspect._empty
        and param.kind in {param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY}
        and param.name not in kwargs
    ]
    if missing_required:
        raise TypeError(f"Cannot call {candidate.__name__}; missing {missing_required}")
    return candidate(**kwargs)


def standardize_ohlcv(
    df: pd.DataFrame,
    *,
    intraday: bool,
    timezone: str,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    frame = df.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [str(col[0]).lower() for col in frame.columns]

    rename_map = {
        "Open": "open",
        "OPEN": "open",
        "open": "open",
        "High": "high",
        "HIGH": "high",
        "high": "high",
        "Low": "low",
        "LOW": "low",
        "low": "low",
        "Close": "close",
        "CLOSE": "close",
        "close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
        "VOLUME": "volume",
        "volume": "volume",
        "DATE": "date",
        "Date": "date",
        "date": "date",
        "Datetime": "datetime",
        "datetime": "datetime",
        "TIMESTAMP": "datetime",
        "timestamp": "datetime",
    }
    frame = frame.rename(columns={col: rename_map.get(col, col) for col in frame.columns})

    if not isinstance(frame.index, pd.DatetimeIndex):
        datetime_column = None
        for candidate in ("datetime", "date"):
            if candidate in frame.columns:
                datetime_column = candidate
                break
        if datetime_column is None:
            raise ValueError("OHLCV dataframe must have a DatetimeIndex or datetime/date column")
        frame.index = pd.to_datetime(frame.pop(datetime_column))

    frame.index = _normalize_index(frame.index, timezone=timezone, intraday=intraday)
    required = ["open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"OHLCV dataframe missing columns: {missing}")

    frame = frame[required].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=["open", "high", "low", "close"])
    frame["volume"] = frame["volume"].fillna(0.0)
    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    return frame


def _normalize_index(index: pd.DatetimeIndex, *, timezone: str, intraday: bool) -> pd.DatetimeIndex:
    dt_index = pd.DatetimeIndex(index)
    if intraday:
        if dt_index.tz is None:
            dt_index = dt_index.tz_localize(timezone, nonexistent="shift_forward", ambiguous="NaT")
        else:
            dt_index = dt_index.tz_convert(timezone)
        return dt_index.tz_localize(None)
    return pd.DatetimeIndex(dt_index.date)


def _safe_name(value: str) -> str:
    return value.replace(".", "_").replace("/", "_").replace(" ", "_")

