from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PathConfig:
    root: Path = PROJECT_ROOT
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"
    artifact_dir: Path = PROJECT_ROOT / "artifacts"
    model_artifact_dir: Path = PROJECT_ROOT / "artifacts" / "models"
    report_dir: Path = PROJECT_ROOT / "reports"

    def ensure(self) -> None:
        for path in (
            self.raw_data_dir,
            self.processed_data_dir,
            self.artifact_dir,
            self.model_artifact_dir,
            self.report_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class MarketConfig:
    symbol: str = "RELIANCE"
    ticker: str = "RELIANCE.NS"
    series: str = "EQ"
    interval: str = "5m"
    timezone: str = "Asia/Kolkata"
    intraday_source: str = "jugaad"
    intraday_fallback_sources: tuple[str, ...] = ("openchart",)
    daily_source: str = "yfinance"
    local_intraday_path: Path | None = None
    local_intraday_pattern: str = "*.csv"
    lookback_days: int = 365

    @property
    def default_start(self) -> date:
        return date.today() - timedelta(days=self.lookback_days)

    @property
    def default_end(self) -> date:
        return date.today()


@dataclass(frozen=True)
class LabelConfig:
    horizon: int = 15
    target_pct: float = 0.005
    stop_pct: float = 0.003


@dataclass(frozen=True)
class NormalizerConfig:
    window: int = 200
    min_periods: int = 200
    epsilon: float = 1e-12


@dataclass(frozen=True)
class FeatureConfig:
    lag_periods: tuple[int, ...] = (1, 5, 10)
    volume_confirm_window: int = 20
    include_daily_context: bool = True


@dataclass(frozen=True)
class SplitConfig:
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15

    def __post_init__(self) -> None:
        total = self.train_ratio + self.validation_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-9:
            raise ValueError("Split ratios must sum to 1.0")


@dataclass(frozen=True)
class SignalConfig:
    confidence_threshold: float = 0.65
    volume_multiplier: float = 1.5


@dataclass(frozen=True)
class BacktestConfig:
    brokerage_rate: float = 0.0003
    slippage_rate: float = 0.0005
    stt_sell_rate: float = 0.00025
    session_start: str = "09:15"
    first_trade_time: str = "09:20"
    session_end: str = "15:30"
    last_entry_time: str = "15:20"
    bars_per_trading_day: int = 75
    starting_equity: float = 1.0


@dataclass(frozen=True)
class PipelineConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    market: MarketConfig = field(default_factory=MarketConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    normalizer: NormalizerConfig = field(default_factory=NormalizerConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    signals: SignalConfig = field(default_factory=SignalConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for group in ("paths",):
            for key, value in data[group].items():
                data[group][key] = str(value)
        return data
