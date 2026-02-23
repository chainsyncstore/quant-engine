from __future__ import annotations

from datetime import datetime, timezone

import pytest

from quant_v2.contracts import PortfolioSnapshot, RiskSnapshot, StrategySignal


def test_strategy_signal_actionable_for_buy_and_sell() -> None:
    buy = StrategySignal(
        symbol="BTCUSDT",
        timeframe="1h",
        horizon_bars=4,
        signal="BUY",
        confidence=0.74,
    )
    hold = StrategySignal(
        symbol="BTCUSDT",
        timeframe="1h",
        horizon_bars=4,
        signal="HOLD",
        confidence=0.50,
    )

    assert buy.actionable is True
    assert hold.actionable is False


def test_strategy_signal_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ValueError):
        StrategySignal(
            symbol="ETHUSDT",
            timeframe="1h",
            horizon_bars=1,
            signal="SELL",
            confidence=1.2,
        )


def test_risk_snapshot_rejects_invalid_net_exposure() -> None:
    with pytest.raises(ValueError):
        RiskSnapshot(
            gross_exposure_frac=0.30,
            net_exposure_frac=0.40,
            max_drawdown_frac=0.10,
            risk_budget_used_frac=0.25,
        )


def test_portfolio_snapshot_symbol_count() -> None:
    snap = PortfolioSnapshot(
        timestamp=datetime.now(timezone.utc),
        equity_usd=12500.0,
        open_positions={"BTCUSDT": 0.12, "ETHUSDT": -0.08},
    )

    assert snap.symbol_count == 2
