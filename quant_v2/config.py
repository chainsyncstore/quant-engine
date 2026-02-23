"""Runtime defaults for the v2 score-first rollout."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

_DEFAULT_UNIVERSE = (
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "LINKUSDT",
    "AVAXUSDT",
    "LTCUSDT",
)


@dataclass(frozen=True)
class UniverseConfig:
    """Universe and timeframe defaults for the initial v2 release."""

    symbols: tuple[str, ...] = _DEFAULT_UNIVERSE
    anchor_interval: str = "1h"
    context_intervals: tuple[str, ...] = ("4h",)
    phase2_symbol_cap: int = 14

    def validate(self) -> None:
        if not self.symbols:
            raise ValueError("Universe cannot be empty")
        if self.anchor_interval in self.context_intervals:
            raise ValueError("Anchor interval must not appear in context intervals")
        if self.phase2_symbol_cap < len(self.symbols):
            raise ValueError("Phase 2 symbol cap cannot be smaller than Phase 1 universe")


@dataclass(frozen=True)
class DeploymentConfig:
    """Rollout and runtime deployment defaults."""

    split_workers: bool = True
    conservative_cutover: bool = True
    canary_live_risk_cap_frac: float = 0.15

    def validate(self) -> None:
        if not 0.0 < self.canary_live_risk_cap_frac <= 1.0:
            raise ValueError("Canary risk cap fraction must be within (0, 1]")


@dataclass(frozen=True)
class DependencyPolicy:
    """Dependency policy for the v2 foundation."""

    approved_phase1: tuple[str, ...] = ("duckdb", "statsmodels", "orjson")
    optional_phase1: tuple[str, ...] = ("redis",)


@dataclass(frozen=True)
class RuntimeProfile:
    """Resolved v2 runtime profile."""

    project_root: Path
    model_registry_root: Path
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    dependency_policy: DependencyPolicy = field(default_factory=DependencyPolicy)

    def validate(self) -> None:
        self.universe.validate()
        self.deployment.validate()


@lru_cache(maxsize=1)
def get_runtime_profile() -> RuntimeProfile:
    """Return singleton runtime defaults for v2 modules."""

    project_root = Path(__file__).resolve().parents[1]
    registry_root = Path(
        os.getenv(
            "BOT_MODEL_REGISTRY_ROOT",
            str(project_root / "models" / "production_v2" / "registry"),
        )
    ).expanduser()

    profile = RuntimeProfile(
        project_root=project_root,
        model_registry_root=registry_root,
    )
    profile.validate()
    return profile


def default_universe_symbols() -> tuple[str, ...]:
    """Expose the default v2 tradable universe."""

    return get_runtime_profile().universe.symbols
