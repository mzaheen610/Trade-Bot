from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, time
from typing import Any

import pandas as pd

from config import BacktestConfig, LabelConfig


@dataclass(frozen=True)
class Trade:
    entry_time: datetime
    exit_time: datetime
    direction: int
    entry_price: float
    exit_price: float
    gross_pnl: float
    costs: float
    net_pnl: float
    return_pct: float
    bars_held: int
    exit_reason: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["side"] = "LONG" if self.direction == 1 else "SHORT"
        return data


@dataclass(frozen=True)
class BacktestResult:
    trades: pd.DataFrame
    equity_curve: pd.DataFrame


@dataclass
class _Position:
    direction: int
    entry_time: datetime
    entry_index: int
    entry_price: float
    target_price: float
    stop_price: float


class BacktestEngine:
    def __init__(self, *, config: BacktestConfig, labels: LabelConfig) -> None:
        self.config = config
        self.labels = labels
        self.session_start = _parse_time(config.session_start)
        self.first_trade_time = _parse_time(config.first_trade_time)
        self.session_end = _parse_time(config.session_end)
        self.last_entry_time = _parse_time(config.last_entry_time)
        self.close_bar_time = time(15, 25)

    def run(self, signal_frame: pd.DataFrame) -> BacktestResult:
        required = {"open", "high", "low", "close", "signal"}
        missing = required.difference(signal_frame.columns)
        if missing:
            raise ValueError(f"Backtest frame missing columns: {sorted(missing)}")

        frame = signal_frame.sort_index().copy()
        frame["signal"] = frame["signal"].fillna(0).astype(int)
        trades: list[Trade] = []
        equity = self.config.starting_equity
        equity_points: list[dict[str, Any]] = []
        position: _Position | None = None

        for idx in range(1, len(frame)):
            previous = frame.iloc[idx - 1]
            row = frame.iloc[idx]
            previous_time = frame.index[idx - 1].to_pydatetime()
            current_time = frame.index[idx].to_pydatetime()

            if position is not None and previous["signal"] == -position.direction:
                trade = self._exit_position(
                    position,
                    exit_time=current_time,
                    exit_index=idx,
                    raw_exit_price=float(row["open"]),
                    reason="opposite_signal",
                )
                trades.append(trade)
                equity *= 1.0 + trade.return_pct
                position = None

            if (
                position is None
                and previous["signal"] in (1, -1)
                and _same_session(previous_time, current_time)
                and self._can_enter(current_time)
            ):
                position = self._enter_position(
                    direction=int(previous["signal"]),
                    entry_time=current_time,
                    entry_index=idx,
                    raw_entry_price=float(row["open"]),
                )

            if position is not None:
                exit_price, reason = self._intrabar_exit(position, row)
                if exit_price is None and idx - position.entry_index >= self.labels.horizon:
                    exit_price = float(row["close"])
                    reason = "timeout"
                if exit_price is None and self._is_session_close_bar(current_time):
                    exit_price = float(row["close"])
                    reason = "session_close"

                if exit_price is not None:
                    trade = self._exit_position(
                        position,
                        exit_time=current_time,
                        exit_index=idx,
                        raw_exit_price=exit_price,
                        reason=reason or "unknown",
                    )
                    trades.append(trade)
                    equity *= 1.0 + trade.return_pct
                    position = None

            equity_points.append({"timestamp": current_time, "equity": equity})

        if position is not None:
            final_time = frame.index[-1].to_pydatetime()
            final_row = frame.iloc[-1]
            trade = self._exit_position(
                position,
                exit_time=final_time,
                exit_index=len(frame) - 1,
                raw_exit_price=float(final_row["close"]),
                reason="end_of_data",
            )
            trades.append(trade)
            equity *= 1.0 + trade.return_pct
            equity_points.append({"timestamp": final_time, "equity": equity})

        trades_df = pd.DataFrame([trade.to_dict() for trade in trades])
        equity_df = pd.DataFrame(equity_points)
        if not equity_df.empty:
            equity_df = equity_df.drop_duplicates("timestamp", keep="last").set_index("timestamp")
        return BacktestResult(trades=trades_df, equity_curve=equity_df)

    def _enter_position(
        self,
        *,
        direction: int,
        entry_time: datetime,
        entry_index: int,
        raw_entry_price: float,
    ) -> _Position:
        entry_price = _apply_slippage(
            raw_entry_price,
            direction=direction,
            is_entry=True,
            slippage_rate=self.config.slippage_rate,
        )
        if direction == 1:
            target_price = entry_price * (1.0 + self.labels.target_pct)
            stop_price = entry_price * (1.0 - self.labels.stop_pct)
        else:
            target_price = entry_price * (1.0 - self.labels.target_pct)
            stop_price = entry_price * (1.0 + self.labels.stop_pct)
        return _Position(
            direction=direction,
            entry_time=entry_time,
            entry_index=entry_index,
            entry_price=entry_price,
            target_price=target_price,
            stop_price=stop_price,
        )

    def _exit_position(
        self,
        position: _Position,
        *,
        exit_time: datetime,
        exit_index: int,
        raw_exit_price: float,
        reason: str,
    ) -> Trade:
        exit_price = _apply_slippage(
            raw_exit_price,
            direction=position.direction,
            is_entry=False,
            slippage_rate=self.config.slippage_rate,
        )
        gross_pnl = (exit_price - position.entry_price) * position.direction
        entry_turnover = abs(position.entry_price)
        exit_turnover = abs(exit_price)
        sell_turnover = exit_turnover if position.direction == 1 else entry_turnover
        costs = (
            (entry_turnover + exit_turnover) * self.config.brokerage_rate
            + sell_turnover * self.config.stt_sell_rate
        )
        net_pnl = gross_pnl - costs
        return_pct = net_pnl / entry_turnover if entry_turnover else 0.0
        return Trade(
            entry_time=position.entry_time,
            exit_time=exit_time,
            direction=position.direction,
            entry_price=position.entry_price,
            exit_price=exit_price,
            gross_pnl=gross_pnl,
            costs=costs,
            net_pnl=net_pnl,
            return_pct=return_pct,
            bars_held=exit_index - position.entry_index,
            exit_reason=reason,
        )

    def _intrabar_exit(self, position: _Position, row: pd.Series) -> tuple[float | None, str | None]:
        high = float(row["high"])
        low = float(row["low"])
        if position.direction == 1:
            stop_hit = low <= position.stop_price
            target_hit = high >= position.target_price
            if stop_hit:
                return position.stop_price, "stop"
            if target_hit:
                return position.target_price, "target"
        else:
            stop_hit = high >= position.stop_price
            target_hit = low <= position.target_price
            if stop_hit:
                return position.stop_price, "stop"
            if target_hit:
                return position.target_price, "target"
        return None, None

    def _can_enter(self, timestamp: datetime) -> bool:
        current = timestamp.time()
        return self.first_trade_time <= current <= self.last_entry_time

    def _is_session_close_bar(self, timestamp: datetime) -> bool:
        current = timestamp.time()
        return self.close_bar_time <= current < self.session_end


def _apply_slippage(
    price: float,
    *,
    direction: int,
    is_entry: bool,
    slippage_rate: float,
) -> float:
    if direction == 1:
        return price * (1.0 + slippage_rate) if is_entry else price * (1.0 - slippage_rate)
    return price * (1.0 - slippage_rate) if is_entry else price * (1.0 + slippage_rate)


def _parse_time(value: str) -> time:
    return datetime.strptime(value, "%H:%M").time()


def _same_session(left: datetime, right: datetime) -> bool:
    return left.date() == right.date()

