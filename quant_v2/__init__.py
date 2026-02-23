"""Foundational modules for the v2 multi-symbol trading stack."""

from quant_v2.config import default_universe_symbols, get_runtime_profile
from quant_v2.contracts import (
    ExecutionIntent,
    OrderPlan,
    PortfolioSnapshot,
    RiskSnapshot,
    StrategySignal,
)
from quant_v2.execution import InMemoryExecutionService
from quant_v2.model_registry import ActiveModelPointer, ModelRegistry, ModelVersionRecord
from quant_v2.research import build_report_from_path

__all__ = [
    "InMemoryExecutionService",
    "ExecutionIntent",
    "ModelRegistry",
    "ActiveModelPointer",
    "ModelVersionRecord",
    "OrderPlan",
    "PortfolioSnapshot",
    "RiskSnapshot",
    "StrategySignal",
    "build_report_from_path",
    "default_universe_symbols",
    "get_runtime_profile",
]
