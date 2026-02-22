"""
Central configuration for the Quant Research Engine.

All tuneable parameters live here. Loaded once at import time.
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env from project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# API Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CapitalAPIConfig:
    """Legacy Capital.com REST API settings (deprecated, kept for compatibility)."""

    api_key: str = field(default_factory=lambda: os.getenv("CAPITAL_API_KEY", ""))
    password: str = field(default_factory=lambda: os.getenv("CAPITAL_PASSWORD", ""))
    identifier: str = field(default_factory=lambda: os.getenv("CAPITAL_IDENTIFIER", ""))
    base_url: str = field(
        default_factory=lambda: os.getenv(
            "CAPITAL_API_URL", "https://demo-api-capital.backend-capital.com"
        )
    )
    epic: str = "EURUSD"
    resolution: str = "MINUTE"
    max_bars_per_request: int = 1000
    rate_limit_per_sec: int = 10


@dataclass(frozen=True)
class BinanceAPIConfig:
    """Binance Futures REST API connection settings."""

    api_key: str = field(default_factory=lambda: os.getenv("BINANCE_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.getenv("BINANCE_API_SECRET", ""))
    base_url: str = "https://fapi.binance.com"
    testnet_url: str = "https://testnet.binancefuture.com"
    symbol: str = "BTCUSDT"
    interval: str = "1h"
    max_bars_per_request: int = 1500
    leverage: int = 1                  # 1x leverage (no margin amplification)
    margin_type: str = "ISOLATED"      # ISOLATED safer than CROSS
    recv_window: int = 5000            # Request validity window (ms)


# ---------------------------------------------------------------------------
# Session Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SessionConfig:
    """Legacy FX trading session windows (UTC hours)."""

    # (start_hour, start_minute, end_hour, end_minute)
    london: Tuple[int, int, int, int] = (8, 0, 16, 30)
    new_york: Tuple[int, int, int, int] = (13, 0, 21, 0)

    @property
    def combined_start_hour(self) -> int:  # noqa: D401
        return min(self.london[0], self.new_york[0])

    @property
    def combined_end_hour(self) -> int:  # noqa: D401
        return max(self.london[2], self.new_york[2])

    @property
    def combined_end_minute(self) -> int:  # noqa: D401
        return max(self.london[3], self.new_york[3])


# ---------------------------------------------------------------------------
# Research Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ResearchConfig:
    """Core research parameters."""

    # Crypto-only runtime mode.
    mode: str = "crypto"

    # Spread / transaction cost
    # FX: 0.8 pips = 0.00008 price units
    # Crypto: 0.04% taker fee per side = 0.08% round trip (set dynamically)
    spread_pips: float = 0.0
    spread_price: float = 0.0  # For crypto, computed dynamically from fee_rate
    taker_fee_rate: float = 0.0004  # 0.04% per side (Binance default)

    # Pessimistic Execution (Stop Loss)
    stop_loss_pips: float = 0.0
    stop_loss_pct: float = 0.02  # 2% stop loss for crypto

    # Horizons to evaluate (in bars: 1H bars for crypto)
    horizons: List[int] = field(default_factory=lambda: [1, 4, 12])

    # Walk-forward windows (in bars)
    # Crypto 1H: 2000 bars = ~83 days train, 500 bars = ~21 days test
    wf_train_bars: int = 2_000
    wf_test_bars: int = 500
    wf_step_bars: int = 500
    wf_calibration_frac: float = 0.20

    # Minimum data requirement
    min_total_bars: int = 8_000  # ~333 days of 1H bars

    # GMM regime count
    n_regimes: int = 5

    # Feature budget
    max_features: int = 90

    # Model hyperparameters
    lgbm_n_estimators: int = 500
    lgbm_max_depth: int = 6
    lgbm_learning_rate: float = 0.05
    lgbm_subsample: float = 0.8
    lgbm_colsample_bytree: float = 0.8
    lgbm_min_child_samples: int = 50

    # Monte Carlo
    mc_n_simulations: int = 10_000

    # Threshold sweep
    threshold_min: float = 0.50
    threshold_max: float = 0.80
    threshold_step: float = 0.05

    # Dead zone for ternary labeling
    # Crypto: percentage-based (0.10% of price)
    dead_zone_pct: float = 0.0010
    dead_zone_pips: float = 0.0  # unused in crypto mode
    dead_zone_price: float = 0.0  # computed dynamically in crypto mode


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PathConfig:
    """Project directory paths."""

    root: Path = _PROJECT_ROOT
    datasets_raw: Path = field(default_factory=lambda: _PROJECT_ROOT / "datasets" / "raw")
    datasets_snapshots: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "datasets" / "snapshots"
    )
    experiments: Path = field(default_factory=lambda: _PROJECT_ROOT / "experiments")
    models: Path = field(default_factory=lambda: _PROJECT_ROOT / "models")

    def ensure_dirs(self) -> None:
        """Create all output directories if they don't exist."""
        for p in [self.datasets_raw, self.datasets_snapshots, self.experiments, self.models]:
            p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Singleton accessors
# ---------------------------------------------------------------------------
_api_cfg: CapitalAPIConfig | None = None
_binance_cfg: BinanceAPIConfig | None = None
_session_cfg: SessionConfig | None = None
_research_cfg: ResearchConfig | None = None
_path_cfg: PathConfig | None = None


def get_api_config() -> CapitalAPIConfig:
    global _api_cfg
    if _api_cfg is None:
        _api_cfg = CapitalAPIConfig()
    return _api_cfg


def get_binance_config() -> BinanceAPIConfig:
    global _binance_cfg
    if _binance_cfg is None:
        _binance_cfg = BinanceAPIConfig()
    return _binance_cfg


def get_session_config() -> SessionConfig:
    global _session_cfg
    if _session_cfg is None:
        _session_cfg = SessionConfig()
    return _session_cfg


def get_research_config() -> ResearchConfig:
    global _research_cfg
    if _research_cfg is None:
        _research_cfg = ResearchConfig()
    return _research_cfg


def get_path_config() -> PathConfig:
    global _path_cfg
    if _path_cfg is None:
        _path_cfg = PathConfig()
        _path_cfg.ensure_dirs()
    return _path_cfg
