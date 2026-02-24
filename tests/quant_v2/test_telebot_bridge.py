from __future__ import annotations

import asyncio

from dataclasses import replace

import pytest

from quant_v2.contracts import StrategySignal
from quant_v2.execution.service import InMemoryExecutionService, RoutedExecutionService, SessionRequest
from quant_v2.monitoring.kill_switch import MonitoringSnapshot
from quant_v2.telebot.bridge import (
    V2ExecutionBridge,
    convert_legacy_signal_payload,
    format_portfolio_snapshot,
)


def test_v2_execution_bridge_session_lifecycle_and_stats() -> None:
    service = InMemoryExecutionService()
    bridge = V2ExecutionBridge(
        service,
        default_strategy_profile="core_v2",
        default_universe=("BTCUSDT", "ETHUSDT"),
    )

    started = asyncio.run(bridge.start_session(user_id=55, live=False))
    assert started is True
    assert bridge.is_running(55) is True

    stats_text = bridge.build_stats_text(55, mode_label="PAPER")
    assert stats_text is not None
    assert "Portfolio Stats (v2)" in stats_text
    assert "Open symbols" in stats_text

    stopped = asyncio.run(bridge.stop_session(55))
    assert stopped is True
    assert bridge.is_running(55) is False


def test_v2_execution_bridge_routes_signals_via_service() -> None:
    service = RoutedExecutionService()
    bridge = V2ExecutionBridge(service, default_universe=("BTCUSDT",))

    assert asyncio.run(bridge.start_session(user_id=56, live=False)) is True
    assert bridge.get_session_mode(56) == "paper"

    signal = StrategySignal(
        symbol="BTCUSDT",
        timeframe="1h",
        horizon_bars=4,
        signal="BUY",
        confidence=0.8,
    )
    results = asyncio.run(
        bridge.route_signals(
            56,
            signals=(signal,),
            prices={"BTCUSDT": 50000.0},
        )
    )

    assert len(results) == 1
    assert results[0].symbol == "BTCUSDT"
    assert results[0].side == "BUY"


def test_v2_execution_bridge_exposes_kill_switch_monitoring_state() -> None:
    service = RoutedExecutionService()
    bridge = V2ExecutionBridge(service, default_universe=("BTCUSDT",))

    assert asyncio.run(bridge.start_session(user_id=57, live=False)) is True

    evaluation = bridge.set_monitoring_snapshot(
        57,
        MonitoringSnapshot(feature_drift_alert=True),
    )
    assert evaluation is not None
    assert evaluation.pause_trading is True
    assert "feature_drift" in evaluation.reasons

    signal = StrategySignal(
        symbol="BTCUSDT",
        timeframe="1h",
        horizon_bars=4,
        signal="BUY",
        confidence=0.8,
    )
    blocked = asyncio.run(
        bridge.route_signals(
            57,
            signals=(signal,),
            prices={"BTCUSDT": 50000.0},
            monitoring_snapshot=MonitoringSnapshot(feature_drift_alert=True),
        )
    )
    assert blocked == ()

    latest = bridge.get_kill_switch_evaluation(57)
    assert latest is not None
    assert latest.pause_trading is True


def test_v2_execution_bridge_exposes_execution_diagnostics() -> None:
    service = RoutedExecutionService()
    bridge = V2ExecutionBridge(service, default_universe=("BTCUSDT",))

    assert asyncio.run(bridge.start_session(user_id=58, live=False)) is True

    signal = StrategySignal(
        symbol="BTCUSDT",
        timeframe="1h",
        horizon_bars=4,
        signal="BUY",
        confidence=0.8,
    )
    routed = asyncio.run(
        bridge.route_signals(
            58,
            signals=(signal,),
            prices={"BTCUSDT": 50000.0},
        )
    )
    assert len(routed) == 1

    diagnostics = bridge.get_execution_diagnostics(58)
    assert diagnostics is not None
    assert diagnostics.total_orders == 1
    assert diagnostics.accepted_orders == 1
    assert diagnostics.rejected_orders == 0


def test_v2_execution_bridge_sync_positions_via_service() -> None:
    service = RoutedExecutionService()
    bridge = V2ExecutionBridge(service, default_universe=("BTCUSDT", "ETHUSDT"))

    assert asyncio.run(bridge.start_session(user_id=581, live=False)) is True
    restored = asyncio.run(
        bridge.sync_positions(
            581,
            target_positions={"BTCUSDT": 1.0, "ETHUSDT": -2.0},
            prices={"BTCUSDT": 100.0, "ETHUSDT": 200.0},
        )
    )
    assert len(restored) == 2
    assert all(item.accepted for item in restored)

    snapshot = service.get_portfolio_snapshot(581)
    assert snapshot is not None
    assert float(snapshot.open_positions.get("BTCUSDT", 0.0)) == 1.0
    assert float(snapshot.open_positions.get("ETHUSDT", 0.0)) == -2.0


def test_v2_execution_bridge_sets_and_gets_lifecycle_rules() -> None:
    service = RoutedExecutionService()
    bridge = V2ExecutionBridge(service, default_universe=("BTCUSDT", "ETHUSDT"))

    assert asyncio.run(bridge.start_session(user_id=582, live=False)) is True

    rules = bridge.set_lifecycle_rules(
        582,
        auto_close_horizon_bars=4,
        stop_loss_pct=0.02,
    )
    assert rules is not None
    assert rules.auto_close_horizon_bars == 4
    assert rules.stop_loss_pct == 0.02

    fetched = bridge.get_lifecycle_rules(582)
    assert fetched is not None
    assert fetched.auto_close_horizon_bars == 4
    assert fetched.stop_loss_pct == 0.02


def test_v2_execution_bridge_reset_session_state_for_demo_session() -> None:
    service = InMemoryExecutionService()
    bridge = V2ExecutionBridge(service)

    assert asyncio.run(bridge.start_session(user_id=59, live=False)) is True
    paused_eval = bridge.set_monitoring_snapshot(
        59,
        MonitoringSnapshot(feature_drift_alert=True),
    )
    assert paused_eval is not None
    assert paused_eval.pause_trading is True

    assert bridge.reset_session_state(59) is True

    evaluation = bridge.get_kill_switch_evaluation(59)
    assert evaluation is not None
    assert evaluation.pause_trading is False


def test_v2_execution_bridge_reset_session_state_rejects_live_session() -> None:
    service = InMemoryExecutionService()
    bridge = V2ExecutionBridge(service)

    assert asyncio.run(bridge.start_session(user_id=60, live=True)) is True
    assert bridge.get_session_mode(60) == "live"
    assert bridge.reset_session_state(60) is False


def test_v2_execution_bridge_reset_session_state_returns_false_when_unsupported() -> None:
    class NoResetService:
        async def start_session(self, request):
            return True

        async def stop_session(self, user_id: int) -> bool:
            return True

        def is_running(self, user_id: int) -> bool:
            return False

        def get_portfolio_snapshot(self, user_id: int):
            return None

        def get_active_count(self) -> int:
            return 0

        def get_session_mode(self, user_id: int) -> str | None:
            return None

        def get_execution_diagnostics(self, user_id: int):
            return None

        def set_monitoring_snapshot(self, user_id: int, snapshot):
            return None

        def get_kill_switch_evaluation(self, user_id: int):
            return None

        async def route_signals(self, user_id: int, *, signals, prices, monitoring_snapshot=None):
            return ()

    bridge = V2ExecutionBridge(NoResetService())
    assert bridge.reset_session_state(777) is False


def test_v2_execution_bridge_sync_positions_raises_when_unsupported() -> None:
    class NoSyncService:
        async def start_session(self, request):
            return True

        async def stop_session(self, user_id: int) -> bool:
            return True

        def is_running(self, user_id: int) -> bool:
            return False

        def get_portfolio_snapshot(self, user_id: int):
            return None

        def get_active_count(self) -> int:
            return 0

        def get_session_mode(self, user_id: int) -> str | None:
            return None

        def get_execution_diagnostics(self, user_id: int):
            return None

        def set_monitoring_snapshot(self, user_id: int, snapshot):
            return None

        def get_kill_switch_evaluation(self, user_id: int):
            return None

        async def route_signals(self, user_id: int, *, signals, prices, monitoring_snapshot=None):
            return ()

    bridge = V2ExecutionBridge(NoSyncService())
    with pytest.raises(RuntimeError, match="does not support manual position sync"):
        asyncio.run(bridge.sync_positions(777, target_positions={"BTCUSDT": 1.0}))


def test_v2_execution_bridge_lifecycle_helpers_return_none_when_unsupported() -> None:
    class NoLifecycleService:
        async def start_session(self, request):
            return True

        async def stop_session(self, user_id: int) -> bool:
            return True

        def is_running(self, user_id: int) -> bool:
            return False

        def get_portfolio_snapshot(self, user_id: int):
            return None

        def get_active_count(self) -> int:
            return 0

        def get_session_mode(self, user_id: int) -> str | None:
            return None

        def get_execution_diagnostics(self, user_id: int):
            return None

        def set_monitoring_snapshot(self, user_id: int, snapshot):
            return None

        def get_kill_switch_evaluation(self, user_id: int):
            return None

        async def route_signals(self, user_id: int, *, signals, prices, monitoring_snapshot=None):
            return ()

    bridge = V2ExecutionBridge(NoLifecycleService())
    assert bridge.set_lifecycle_rules(777, auto_close_horizon_bars=4, stop_loss_pct=0.02) is None
    assert bridge.get_lifecycle_rules(777) is None


def test_convert_legacy_signal_payload_maps_buy_signal() -> None:
    mapped = convert_legacy_signal_payload(
        {
            "signal": "BUY",
            "close_price": 50500.0,
            "probability": 0.77,
            "horizon": 6,
            "reason": "edge_up",
        },
        default_symbol="BTCUSDT",
    )

    assert mapped is not None
    signal, prices = mapped
    assert signal.symbol == "BTCUSDT"
    assert signal.signal == "BUY"
    assert signal.horizon_bars == 6
    assert signal.confidence == 0.77
    assert prices == {"BTCUSDT": 50500.0}


def test_convert_legacy_signal_payload_ignores_non_actionable() -> None:
    assert (
        convert_legacy_signal_payload(
            {
                "signal": "HOLD",
                "close_price": 50500.0,
                "probability": 0.5,
            },
            default_symbol="BTCUSDT",
        )
        is None
    )


def test_format_portfolio_snapshot_handles_missing_risk() -> None:
    service = InMemoryExecutionService()
    asyncio.run(service.start_session(SessionRequest(user_id=99, live=False)))
    snap = service.get_portfolio_snapshot(99)
    assert snap is not None

    # force a snapshot without risk for rendering edge case
    no_risk = replace(snap, risk=None)
    text = format_portfolio_snapshot(no_risk, mode_label="LIVE")
    assert "LIVE" in text
    assert "Risk Budget Used" in text


def test_format_portfolio_snapshot_includes_notional_breakdown() -> None:
    service = InMemoryExecutionService()
    asyncio.run(service.start_session(SessionRequest(user_id=100, live=False)))
    snap = service.get_portfolio_snapshot(100)
    assert snap is not None

    custom = replace(
        snap,
        open_positions={"BTCUSDT": 0.01, "ETHUSDT": -0.02},
        symbol_notional_usd={"BTCUSDT": 150.0, "ETHUSDT": 200.0},
    )
    text = format_portfolio_snapshot(custom, mode_label="PAPER")
    assert "Total Notional" in text
    assert "Cash Available" in text
    assert "Avg per symbol" in text
    assert "Per-symbol stake" in text
    assert "BTCUSDT" in text
    assert "ETHUSDT" in text


def test_format_portfolio_snapshot_lists_all_symbol_stakes_without_truncation() -> None:
    service = InMemoryExecutionService()
    asyncio.run(service.start_session(SessionRequest(user_id=102, live=False)))
    snap = service.get_portfolio_snapshot(102)
    assert snap is not None

    symbol_notionals = {f"SYM{i}USDT": float(200 - i) for i in range(12)}
    open_positions = {symbol: 0.01 for symbol in symbol_notionals}
    custom = replace(
        snap,
        open_positions=open_positions,
        symbol_notional_usd=symbol_notionals,
    )
    text = format_portfolio_snapshot(custom, mode_label="PAPER")

    for symbol in symbol_notionals:
        assert symbol in text


def test_format_portfolio_snapshot_includes_symbol_pnl_when_available() -> None:
    service = InMemoryExecutionService()
    asyncio.run(service.start_session(SessionRequest(user_id=101, live=False)))
    snap = service.get_portfolio_snapshot(101)
    assert snap is not None

    custom = replace(
        snap,
        open_positions={"BTCUSDT": 0.01, "ETHUSDT": -0.02},
        symbol_notional_usd={"BTCUSDT": 150.0, "ETHUSDT": 200.0},
        symbol_pnl_usd={
            "BTCUSDT": 12.5,
            "ETHUSDT": -8.0,
            "XRPUSDT": 18.0,
            "SOLUSDT": 0.5,
            "ADAUSDT": -10.0,
            "LTCUSDT": 6.0,
        },
    )
    text = format_portfolio_snapshot(custom, mode_label="PAPER")
    assert "Top Symbol PnL" in text

    pnl_section = text.split("Top Symbol PnL:\n", 1)[1]
    expected_order = ["XRPUSDT", "BTCUSDT", "LTCUSDT", "SOLUSDT", "ETHUSDT", "ADAUSDT"]
    for symbol in expected_order:
        assert symbol in pnl_section

    first_index = -1
    for symbol in expected_order:
        current_index = pnl_section.index(f"- {symbol}:")
        assert current_index > first_index
        first_index = current_index
