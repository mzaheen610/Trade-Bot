from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from config import MarketConfig, PathConfig

OPENCHART_EQUITY_TOKENS = {
    "RELIANCE": "2885",
}


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
        output_path = self.intraday_path()
        if output_path.exists() and not force_refresh:
            df = pd.read_parquet(output_path)
            return DownloadResult(
                output_path,
                source or self.market.intraday_source,
                len(df),
                refreshed=False,
            )

        start = start or self.market.default_start
        end = end or self.market.default_end

        sources = [source] if source else [
            self.market.intraday_source,
            *self.market.intraday_fallback_sources,
        ]
        sources = _dedupe_sources([candidate for candidate in sources if candidate])
        errors: list[str] = []
        for candidate_source in sources:
            try:
                df = self._download_intraday_from_source(
                    candidate_source,
                    start=start,
                    end=end,
                )
                df = standardize_ohlcv(df, intraday=True, timezone=self.market.timezone)
                if df.empty:
                    raise DataUnavailableError(
                        f"{candidate_source} returned no intraday bars for {self.market.symbol}"
                    )
                df.to_parquet(output_path)
                return DownloadResult(output_path, candidate_source, len(df), refreshed=True)
            except DataUnavailableError as exc:
                errors.append(f"{candidate_source}: {exc}")

        raise DataUnavailableError(
            "No configured intraday source produced 5-minute bars. Tried "
            f"{', '.join(sources)}. Details: " + " | ".join(errors)
        )

    def _download_intraday_from_source(
        self,
        source: str,
        *,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        if source == "openchart":
            return self._download_openchart_intraday(start=start, end=end)
        if source == "jugaad":
            return self._download_jugaad_intraday(start=start, end=end)
        if source == "local-csv":
            return self._load_local_csv_intraday()
        if source == "yfinance-5m":
            return self._download_yfinance_intraday(start=start, end=end)
        raise ValueError(f"Unknown intraday source: {source}")

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

        if self.market.daily_source == "intraday-resample":
            df = self._daily_context_from_intraday()
        else:
            start = start or date(2000, 1, 1)
            end = end or self.market.default_end
            df = self._download_yfinance_daily(start=start, end=end)
        df = standardize_ohlcv(df, intraday=False, timezone=self.market.timezone)
        if df.empty:
            raise DataUnavailableError(
                f"{self.market.daily_source} returned no daily bars for {self.market.ticker}"
            )

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

    def _load_local_csv_intraday(self) -> pd.DataFrame:
        files = self._local_intraday_files()
        if not files:
            raise DataUnavailableError(
                "No local intraday CSV files found. Set --local-data-path to a "
                "CSV file or folder such as data/BANK_NIFTY_data."
            )

        frames = [_read_local_ohlc_csv(path, symbol=self.market.symbol) for path in files]
        frame = pd.concat(frames, axis=0).sort_index()
        frame = frame[~frame.index.duplicated(keep="last")]
        frame = _filter_market_hours(frame)
        if frame.empty:
            raise DataUnavailableError(f"Local CSV files contained no rows for {self.market.symbol}")
        return _resample_ohlcv(frame, self.market.interval)

    def _local_intraday_files(self) -> list[Path]:
        local_path = self.market.local_intraday_path or _default_local_path(self.market.symbol)
        if local_path is None:
            return []
        local_path = Path(local_path)
        if local_path.is_file():
            return [local_path]
        if not local_path.exists():
            return []
        files = sorted(path for path in local_path.glob(self.market.local_intraday_pattern) if path.is_file())
        return _prefer_yearly_files(files)

    def _daily_context_from_intraday(self) -> pd.DataFrame:
        intraday = self.load_intraday()
        return intraday.resample("1D").agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        ).dropna(subset=["open", "high", "low", "close"])

    def _download_jugaad_intraday(self, *, start: date, end: date) -> pd.DataFrame:
        try:
            import jugaad_data.nse as nse
        except ImportError as exc:
            raise DataUnavailableError(
                "jugaad-data is not installed. Install dependencies or use "
                "--intraday-source yfinance-5m for a smoke-test fallback."
            ) from exc

        if self.market.interval.lower() not in {"1d", "d", "day"}:
            if callable(getattr(nse, "stock_df", None)):
                raise DataUnavailableError(
                    "jugaad-data's documented stock_df API returns historical "
                    "EOD stock rows, not 5-minute intraday OHLCV bars."
                )

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
            "intraday OHLCV dataframe function. Use --intraday-source openchart "
            "for NSE charting data, or --intraday-source yfinance-5m only for a "
            "short smoke test."
        )

    def _download_openchart_intraday(self, *, start: date, end: date) -> pd.DataFrame:
        try:
            from openchart import NSEData
        except ImportError as exc:
            raise DataUnavailableError(
                "openchart is not installed. Run `pip install -e .` again, or "
                "use an uploaded historical intraday file."
            ) from exc

        symbol = f"{self.market.symbol}-{self.market.series}"
        try:
            client = NSEData()
            start_dt = datetime.combine(start, datetime.min.time())
            end_dt = datetime.combine(end, datetime.max.time())
            token = OPENCHART_EQUITY_TOKENS.get(self.market.symbol.upper())
            if token:
                return client.historical_direct(
                    token=token,
                    symbol=symbol,
                    symbol_type="Equity",
                    start=start_dt,
                    end=end_dt,
                    interval=self.market.interval,
                )
            return client.historical(symbol, "EQ", start_dt, end_dt, self.market.interval)
        except Exception as exc:
            raise DataUnavailableError(
                f"openchart could not fetch {symbol} {self.market.interval} data. "
                "NSE charting availability varies; try a shorter lookback or use "
                "an uploaded historical intraday Parquet/CSV file."
            ) from exc

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


def _dedupe_sources(sources: list[str]) -> list[str]:
    deduped: list[str] = []
    for source in sources:
        if source not in deduped:
            deduped.append(source)
    return deduped


def _default_local_path(symbol: str) -> Path | None:
    normalized = symbol.upper().replace("_", "")
    if normalized in {"BANKNIFTY", "BNF"}:
        return Path("data") / "BANK_NIFTY_data"
    if normalized in {"NIFTY", "NIFTY50"}:
        return Path("data") / "NIFTY_data"
    return None


def _prefer_yearly_files(files: list[Path]) -> list[Path]:
    yearly = [path for path in files if _has_single_year_suffix(path.stem)]
    return yearly or files


def _has_single_year_suffix(stem: str) -> bool:
    years = re.findall(r"(?:19|20)\d{2}", stem)
    return len(years) == 1 and stem.endswith(years[0])


def _read_local_ohlc_csv(path: Path, *, symbol: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    rename_map = {
        "Instrument": "instrument",
        "Date": "date",
        "Time": "time",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    frame = frame.rename(columns={column: rename_map.get(column, column) for column in frame.columns})
    if "instrument" in frame.columns and symbol:
        normalized_symbol = symbol.upper().replace("_", "")
        instruments = frame["instrument"].astype(str).str.upper().str.replace("_", "", regex=False)
        if normalized_symbol in set(instruments.unique()):
            frame = frame[instruments == normalized_symbol]

    if {"date", "time"}.issubset(frame.columns):
        timestamps = pd.to_datetime(
            frame["date"].astype(str) + " " + frame["time"].astype(str),
            format="%Y%m%d %H:%M",
            errors="coerce",
        )
    elif "datetime" in frame.columns:
        timestamps = pd.to_datetime(frame["datetime"], errors="coerce")
    else:
        raise ValueError(f"{path} must include Date/Time columns or a datetime column")

    output = frame.copy()
    output.index = timestamps
    output = output[~output.index.isna()]
    output = output.dropna(subset=["open", "high", "low", "close"])
    if "volume" not in output.columns:
        output["volume"] = 1.0
    output = output[["open", "high", "low", "close", "volume"]].apply(
        pd.to_numeric,
        errors="coerce",
    )
    output = output.dropna(subset=["open", "high", "low", "close"])
    output["volume"] = output["volume"].fillna(1.0)
    return output.sort_index()


def _filter_market_hours(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.between_time("09:15", "15:30", inclusive="both")


def _resample_ohlcv(frame: pd.DataFrame, interval: str) -> pd.DataFrame:
    rule = _pandas_interval(interval)
    if rule == "1min":
        return frame.copy()
    resampled = frame.resample(
        rule,
        origin="start_day",
        offset="9h15min",
        label="right",
        closed="right",
    ).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    return resampled.dropna(subset=["open", "high", "low", "close"])


def _pandas_interval(interval: str) -> str:
    normalized = interval.lower().strip()
    match = re.fullmatch(r"(\d+)\s*m(?:in(?:ute)?s?)?", normalized)
    if match:
        return f"{int(match.group(1))}min"
    if normalized in {"minute", "1minute"}:
        return "1min"
    raise ValueError(f"Unsupported local CSV interval: {interval}")


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
