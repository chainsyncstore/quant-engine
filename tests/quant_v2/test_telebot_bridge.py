from __future__ import annotations

import asyncio

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
    from dataclasses import replace

    no_risk = replace(snap, risk=None)
    text = format_portfolio_snapshot(no_risk, mode_label="LIVE")
    assert "LIVE" in text
    assert "Risk Budget Used" in text
