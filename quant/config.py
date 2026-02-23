"""
Central configuration for the Quant Research Engine.

All tuneable parameters live here. Loaded once at import time.
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env from project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


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
# Research Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ResearchConfig:
    """Core research parameters."""

    # Crypto-only runtime mode.
    mode: str = "crypto"

    # Transaction cost assumptions
    # Round-trip taker fee = close * taker_fee_rate * 2
    spread_price: float = 0.0  # Optional fixed overlay for reporting/backward compatibility
    taker_fee_rate: float = 0.0004  # 0.04% per side (Binance default)

    # Pessimistic Execution (Stop Loss)
    stop_loss_pct: float = 0.02  # 2% stop loss for crypto

    # Horizons to evaluate (in bars: 1H bars for crypto)
    horizons: List[int] = field(default_factory=lambda: [1, 4, 12])

    # Walk-forward windows (in bars)
    # Crypto 1H: 2000 bars = ~83 days train, 500 bars = ~21 days test
    wf_train_bars: int = 2_000
    wf_test_bars: int = 500
    wf_step_bars: int = 500
    wf_embargo_bars: int = 24
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

    # Dead zone for ternary labeling (percentage of close)
    dead_zone_pct: float = 0.0010


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
_binance_cfg: BinanceAPIConfig | None = None
_research_cfg: ResearchConfig | None = None
_path_cfg: PathConfig | None = None


def get_binance_config() -> BinanceAPIConfig:
    global _binance_cfg
    if _binance_cfg is None:
        _binance_cfg = BinanceAPIConfig()
    return _binance_cfg


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
