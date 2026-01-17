"""Execution policy definition for the temporary 5%ers competition mode."""

from datetime import time

from config.execution_policies import ExecutionPolicy

COMPETITION_5PERCENTERS: ExecutionPolicy = ExecutionPolicy(
    policy_id="COMPETITION_5PERCENTERS",
    description="Temporary contest profile tuned for 5%ers ranking events.",
    version="1.0",
    label="COMPETITION_5PERCENTERS",
    max_drawdown=0.08,  # 8% lifecycle stop
    daily_drawdown=0.04,  # 4% daily cap
    max_trades_per_day=5,
    max_concurrent_positions=3,
    max_order_notional=0.15,
    min_order_notional=0.02,
    instrument_whitelist=(
        "EURUSD",
        "GBPUSD",
        "USDJPY",
        "AUDCAD",
        "BTCUSD",
        "ETHUSD",
    ),
    trading_window_start=None,  # 24/7 crypto trading
    trading_window_end=None,
    forced_flat_time=None,  # No forced flat for crypto
    metadata={
        "mode": "competition",
        "firm": "5percenters",
        "notes": "Temporary contest profile. Delete after event.",
    },
)

__all__ = ["COMPETITION_5PERCENTERS"]
