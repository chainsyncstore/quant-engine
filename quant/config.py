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
    """Capital.com REST API connection settings."""

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


# ---------------------------------------------------------------------------
# Session Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SessionConfig:
    """Trading session windows (UTC hours)."""

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

    # Spread (pips → price units for EURUSD: 1 pip = 0.0001)
    spread_pips: float = 0.8
    spread_price: float = 0.00008  # 0.8 * 0.0001

    # Horizons to evaluate
    horizons: List[int] = field(default_factory=lambda: [3, 5, 10])

    # Walk-forward windows (in bars = 1-minute candles)
    wf_train_bars: int = 15_000
    wf_test_bars: int = 3_000
    wf_step_bars: int = 3_000
    wf_calibration_frac: float = 0.20  # last 20% of train for calibration

    # Minimum data requirement
    min_total_bars: int = 33_000

    # GMM regime count
    n_regimes: int = 4

    # Feature budget
    max_features: int = 55

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

    # Dead zone for ternary labeling (moves smaller than this are FLAT)
    dead_zone_pips: float = 0.8
    dead_zone_price: float = 0.00008  # 0.8 * 0.0001


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
_session_cfg: SessionConfig | None = None
_research_cfg: ResearchConfig | None = None
_path_cfg: PathConfig | None = None


def get_api_config() -> CapitalAPIConfig:
    global _api_cfg
    if _api_cfg is None:
        _api_cfg = CapitalAPIConfig()
    return _api_cfg


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
