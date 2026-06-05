from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    value = 100.0 - (100.0 / (1.0 + rs))
    value = value.mask((avg_loss == 0.0) & (avg_gain > 0.0), 100.0)
    value = value.mask((avg_gain == 0.0) & (avg_loss > 0.0), 0.0)
    value = value.mask((avg_gain == 0.0) & (avg_loss == 0.0), 50.0)
    return value


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    signal_line = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = line - signal_line
    return pd.DataFrame(
        {
            "macd_line": line,
            "macd_signal": signal_line,
            "macd_hist": histogram,
        },
        index=close.index,
    )


def bollinger_bands(
    close: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    middle = close.rolling(window=window, min_periods=window).mean()
    std = close.rolling(window=window, min_periods=window).std(ddof=0)
    upper = middle + num_std * std
    lower = middle - num_std * std
    width = (upper - lower) / middle.replace(0.0, np.nan)
    percent_b = (close - lower) / (upper - lower).replace(0.0, np.nan)
    return pd.DataFrame(
        {
            "bb_middle_20": middle,
            "bb_upper_20_2": upper,
            "bb_lower_20_2": lower,
            "bb_width_20_2": width,
            "bb_percent_b_20_2": percent_b,
        },
        index=close.index,
    )


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    previous_close = close.shift(1)
    parts = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    )
    return parts.max(axis=1)


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    return true_range(high, low, close).rolling(window=period, min_periods=period).mean()


def anchored_vwap(df: pd.DataFrame) -> pd.Series:
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    traded_value = typical_price * df["volume"]
    sessions = pd.Series(df.index.date, index=df.index)
    cumulative_value = traded_value.groupby(sessions).cumsum()
    cumulative_volume = df["volume"].groupby(sessions).cumsum()
    return cumulative_value / cumulative_volume.replace(0.0, np.nan)


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume).cumsum()


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = true_range(high, low, close)
    atr_value = tr.rolling(window=period, min_periods=period).mean()
    plus_di = 100.0 * pd.Series(plus_dm, index=high.index).rolling(period).mean() / atr_value
    minus_di = 100.0 * pd.Series(minus_dm, index=high.index).rolling(period).mean() / atr_value
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return dx.rolling(window=period, min_periods=period).mean()
