from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from quant_v2.contracts import StrategySignal
from quant_v2.execution.adapters import ExecutionResult
from quant_v2.execution.planner import PlannerConfig
from quant_v2.execution.service import (
    InMemoryExecutionService,
    RoutedExecutionService,
    SessionRequest,
)
from quant_v2.monitoring.kill_switch import MonitoringSnapshot
from quant_v2.portfolio.risk_policy import PortfolioRiskPolicy


def test_in_memory_execution_service_lifecycle() -> None:
    service = InMemoryExecutionService()
    req = SessionRequest(
        user_id=101,
        live=False,
        strategy_profile="core_v2",
        universe=("BTCUSDT", "ETHUSDT"),
    )

    assert asyncio.run(service.start_session(req)) is True
    assert asyncio.run(service.start_session(req)) is False
    assert service.is_running(101) is True
    assert service.get_active_count() == 1
    assert service.get_session_mode(101) == "paper"

    snap = service.get_portfolio_snapshot(101)
    assert snap is not None
    assert snap.equity_usd == 10_000.0

    monitoring_eval = service.set_monitoring_snapshot(
        101,
        MonitoringSnapshot(feature_drift_alert=True),
    )
    assert monitoring_eval.pause_trading is True
    latest_eval = service.get_kill_switch_evaluation(101)
    assert latest_eval is not None
    assert latest_eval.pause_trading is True

    diagnostics = service.get_execution_diagnostics(101)
    assert diagnostics is not None
    assert diagnostics.total_orders == 0

    assert asyncio.run(service.stop_session(101)) is True
    assert asyncio.run(service.stop_session(101)) is False
    assert service.is_running(101) is False
    assert service.get_active_count() == 0


def test_in_memory_execution_service_reset_session_state_reinitializes_demo_state() -> None:
    service = InMemoryExecutionService()
    req = SessionRequest(user_id=109, live=False)
    assert asyncio.run(service.start_session(req)) is True

    paused_eval = service.set_monitoring_snapshot(
        109,
        MonitoringSnapshot(feature_drift_alert=True),
    )
    assert paused_eval.pause_trading is True

    assert service.reset_session_state(109) is True
    evaluation = service.get_kill_switch_evaluation(109)
    assert evaluation is not None
    assert evaluation.pause_trading is False

    snap = service.get_portfolio_snapshot(109)
    assert snap is not None
    assert snap.equity_usd == 10_000.0
    assert snap.open_positions == {}


def test_in_memory_execution_service_reset_session_state_rejects_live_and_missing() -> None:
    service = InMemoryExecutionService()
    assert service.reset_session_state(998) is False

    assert asyncio.run(service.start_session(SessionRequest(user_id=110, live=True))) is True
    assert service.reset_session_state(110) is False


def test_routed_execution_service_demo_uses_paper_adapter_factory() -> None:
    class FakePaperAdapter:
        def get_positions(self):
            return {"BTCUSDT": 0.02}

    service = RoutedExecutionService(
        paper_adapter_factory=FakePaperAdapter,
        live_adapter_factory=lambda request: FakePaperAdapter(),
    )

    req = SessionRequest(user_id=202, live=False)

    assert asyncio.run(service.start_session(req)) is True
    assert service.get_session_mode(202) == "paper"
    snapshot = service.get_portfolio_snapshot(202)
    assert snapshot is not None
    assert snapshot.open_positions == {"BTCUSDT": 0.02}


def test_routed_execution_service_live_routes_request_into_live_factory() -> None:
    captured: dict[str, SessionRequest] = {}

    class FakeLiveAdapter:
        def get_positions(self):
            return {"ETHUSDT": -0.1}

    def live_factory(request: SessionRequest) -> FakeLiveAdapter:
        captured["request"] = request
        return FakeLiveAdapter()

    service = RoutedExecutionService(
        paper_adapter_factory=FakeLiveAdapter,
        live_adapter_factory=live_factory,
    )

    req = SessionRequest(
        user_id=303,
        live=True,
        credentials={"binance_api_key": "k", "binance_api_secret": "s"},
    )

    assert asyncio.run(service.start_session(req)) is True
    assert service.get_session_mode(303) == "live"
    assert captured["request"].credentials["binance_api_key"] == "k"


def test_routed_execution_service_live_session_falls_back_to_paper_when_live_disabled() -> None:
    called = {"live_factory": False}

    class FakePaperAdapter:
        def get_positions(self):
            return {}

    def live_factory(request: SessionRequest):
        called["live_factory"] = True
        return FakePaperAdapter()

    service = RoutedExecutionService(
        paper_adapter_factory=FakePaperAdapter,
        live_adapter_factory=live_factory,
        allow_live_execution=False,
    )

    req = SessionRequest(
        user_id=304,
        live=True,
        credentials={"binance_api_key": "k", "binance_api_secret": "s"},
    )

    assert asyncio.run(service.start_session(req)) is True
    assert called["live_factory"] is False
    assert service.get_session_mode(304) == "paper_shadow"


def test_routed_execution_service_route_signals_executes_orders_and_updates_snapshot() -> None:
    service = RoutedExecutionService()
    req = SessionRequest(user_id=401, live=False)
    assert asyncio.run(service.start_session(req)) is True

    signals = (
        StrategySignal(
            symbol="BTCUSDT",
            timeframe="1h",
            horizon_bars=4,
            signal="BUY",
            confidence=0.76,
            uncertainty=0.10,
        ),
        StrategySignal(
            symbol="ETHUSDT",
            timeframe="1h",
            horizon_bars=4,
            signal="SELL",
            confidence=0.73,
            uncertainty=0.10,
        ),
    )
    prices = {"BTCUSDT": 50000.0, "ETHUSDT": 2500.0}

    first = asyncio.run(service.route_signals(401, signals=signals, prices=prices))
    assert len(first) == 2
    by_symbol = {item.symbol: item for item in first}
    assert by_symbol["BTCUSDT"].side == "BUY"
    assert by_symbol["ETHUSDT"].side == "SELL"

    snap = service.get_portfolio_snapshot(401)
    assert snap is not None
    assert set(snap.open_positions) == {"BTCUSDT", "ETHUSDT"}
    assert snap.risk is not None
    assert snap.risk.gross_exposure_frac > 0.0

    second = asyncio.run(service.route_signals(401, signals=signals, prices=prices))
    assert second == ()


def test_routed_execution_service_hold_only_signals_do_not_flatten_existing_positions() -> None:
    service = RoutedExecutionService()
    req = SessionRequest(user_id=402, live=False)
    assert asyncio.run(service.start_session(req)) is True

    buy_signal = StrategySignal(
        symbol="BTCUSDT",
        timeframe="1h",
        horizon_bars=4,
        signal="BUY",
        confidence=0.80,
    )
    prices = {"BTCUSDT": 50000.0}
    assert asyncio.run(service.route_signals(402, signals=(buy_signal,), prices=prices))

    before = service.get_portfolio_snapshot(402)
    assert before is not None
    before_qty = float(before.open_positions.get("BTCUSDT", 0.0))
    assert before_qty > 0.0

    hold_only = StrategySignal(
        symbol="BTCUSDT",
        timeframe="1h",
        horizon_bars=4,
        signal="HOLD",
        confidence=0.5,
    )
    routed = asyncio.run(service.route_signals(402, signals=(hold_only,), prices=prices))
    assert routed == ()

    after = service.get_portfolio_snapshot(402)
    assert after is not None
    assert float(after.open_positions.get("BTCUSDT", 0.0)) == pytest.approx(before_qty)


def test_routed_execution_service_rejects_adapter_without_get_positions() -> None:
    service = RoutedExecutionService(
        paper_adapter_factory=object,
        live_adapter_factory=lambda request: object(),
    )

    with pytest.raises(TypeError):
        asyncio.run(service.start_session(SessionRequest(user_id=404, live=False)))


def test_routed_execution_service_kill_switch_blocks_orders_and_reports_reasons() -> None:
    service = RoutedExecutionService()
    req = SessionRequest(user_id=405, live=False)
    assert asyncio.run(service.start_session(req)) is True

    signal = StrategySignal(
        symbol="BTCUSDT",
        timeframe="1h",
        horizon_bars=4,
        signal="BUY",
        confidence=0.81,
    )
    prices = {"BTCUSDT": 50000.0}

    routed = asyncio.run(
        service.route_signals(
            405,
            signals=(signal,),
            prices=prices,
            monitoring_snapshot=MonitoringSnapshot(feature_drift_alert=True),
        )
    )
    assert routed == ()

    evaluation = service.get_kill_switch_evaluation(405)
    assert evaluation is not None
    assert evaluation.pause_trading is True
    assert "feature_drift" in evaluation.reasons


def test_routed_execution_service_kill_switch_clears_after_monitoring_recovers() -> None:
    service = RoutedExecutionService()
    req = SessionRequest(user_id=406, live=False)
    assert asyncio.run(service.start_session(req)) is True

    paused_eval = service.set_monitoring_snapshot(
        406,
        MonitoringSnapshot(confidence_collapse_alert=True),
    )
    assert paused_eval.pause_trading is True
    assert "confidence_collapse" in paused_eval.reasons

    resumed_eval = service.set_monitoring_snapshot(406, MonitoringSnapshot())
    assert resumed_eval.pause_trading is False

    signal = StrategySignal(
        symbol="BTCUSDT",
        timeframe="1h",
        horizon_bars=4,
        signal="BUY",
        confidence=0.81,
    )

    routed = asyncio.run(
        service.route_signals(
            406,
            signals=(signal,),
            prices={"BTCUSDT": 50000.0},
        )
    )
    assert len(routed) == 1


def test_routed_execution_service_execution_anomaly_trips_kill_switch_next_cycle() -> None:
    class RejectingAdapter:
        def __init__(self) -> None:
            self._seq = 0

        def get_positions(self):
            return {}

        def place_order(self, plan, *, idempotency_key: str, mark_price: float | None = None):
            self._seq += 1
            return ExecutionResult(
                accepted=False,
                order_id=f"reject-{self._seq}",
                idempotency_key=idempotency_key,
                symbol=plan.symbol,
                side=plan.side,
                requested_qty=plan.quantity,
                filled_qty=0.0,
                avg_price=float(mark_price or 0.0),
                status="rejected",
                created_at=datetime.now(timezone.utc).isoformat(),
                reason="exchange_reject",
            )

    service = RoutedExecutionService(paper_adapter_factory=RejectingAdapter)
    assert asyncio.run(service.start_session(SessionRequest(user_id=407, live=False))) is True

    signal = StrategySignal(
        symbol="BTCUSDT",
        timeframe="1h",
        horizon_bars=4,
        signal="BUY",
        confidence=0.8,
    )

    first = asyncio.run(
        service.route_signals(
            407,
            signals=(signal,),
            prices={"BTCUSDT": 50000.0},
        )
    )
    assert len(first) == 1
    assert first[0].accepted is False

    diagnostics = service.get_execution_diagnostics(407)
    assert diagnostics is not None
    assert diagnostics.total_orders == 1
    assert diagnostics.accepted_orders == 0
    assert diagnostics.rejected_orders == 1
    assert diagnostics.reject_rate == pytest.approx(1.0)

    evaluation = service.get_kill_switch_evaluation(407)
    assert evaluation is not None
    assert evaluation.pause_trading is True
    assert "execution_anomaly" in evaluation.reasons

    blocked = asyncio.run(
        service.route_signals(
            407,
            signals=(signal,),
            prices={"BTCUSDT": 50000.0},
        )
    )
    assert blocked == ()


def test_routed_execution_service_tracks_adverse_slippage_diagnostics() -> None:
    class SlippageAdapter:
        def __init__(self) -> None:
            self._seq = 0
            self._positions: dict[str, float] = {}

        def get_positions(self):
            return dict(self._positions)

        def place_order(self, plan, *, idempotency_key: str, mark_price: float | None = None):
            self._seq += 1
            side_sign = 1.0 if plan.side == "BUY" else -1.0
            self._positions[plan.symbol] = self._positions.get(plan.symbol, 0.0) + (side_sign * plan.quantity)

            mark = float(mark_price or 0.0)
            fill = mark * 1.001 if plan.side == "BUY" else mark * 0.999
            return ExecutionResult(
                accepted=True,
                order_id=f"fill-{self._seq}",
                idempotency_key=idempotency_key,
                symbol=plan.symbol,
                side=plan.side,
                requested_qty=plan.quantity,
                filled_qty=plan.quantity,
                avg_price=fill,
                status="filled",
                created_at=datetime.now(timezone.utc).isoformat(),
            )

    service = RoutedExecutionService(paper_adapter_factory=SlippageAdapter)
    assert asyncio.run(service.start_session(SessionRequest(user_id=408, live=False))) is True

    signal = StrategySignal(
        symbol="BTCUSDT",
        timeframe="1h",
        horizon_bars=4,
        signal="BUY",
        confidence=0.83,
    )

    routed = asyncio.run(
        service.route_signals(
            408,
            signals=(signal,),
            prices={"BTCUSDT": 50000.0},
        )
    )
    assert len(routed) == 1
    assert routed[0].accepted is True

    diagnostics = service.get_execution_diagnostics(408)
    assert diagnostics is not None
    assert diagnostics.total_orders == 1
    assert diagnostics.accepted_orders == 1
    assert diagnostics.rejected_orders == 0
    assert diagnostics.reject_rate == pytest.approx(0.0)
    assert diagnostics.slippage_sample_count == 1
    assert diagnostics.avg_adverse_slippage_bps == pytest.approx(10.0, rel=1e-6)


def test_routed_execution_service_tracks_entry_rebalance_exit_activity() -> None:
    service = RoutedExecutionService(
        risk_policy=PortfolioRiskPolicy(
            max_symbol_exposure_frac=0.10,
            max_gross_exposure_frac=0.20,
            max_net_exposure_frac=0.20,
        )
    )
    assert asyncio.run(service.start_session(SessionRequest(user_id=509, live=False))) is True

    buy_signal = StrategySignal(
        symbol="BTCUSDT",
        timeframe="1h",
        horizon_bars=4,
        signal="BUY",
        confidence=0.90,
    )

    routed_entry = asyncio.run(
        service.route_signals(
            509,
            signals=(buy_signal,),
            prices={"BTCUSDT": 100.0},
            planner_config=PlannerConfig(
                total_risk_budget_frac=0.10,
                max_symbol_exposure_frac=0.10,
                min_confidence=0.0,
            ),
        )
    )
    assert len(routed_entry) == 1

    routed_rebalance = asyncio.run(
        service.route_signals(
            509,
            signals=(buy_signal,),
            prices={"BTCUSDT": 100.0},
            planner_config=PlannerConfig(
                total_risk_budget_frac=0.06,
                max_symbol_exposure_frac=0.10,
                min_confidence=0.0,
            ),
        )
    )
    assert len(routed_rebalance) == 1

    low_conf_signal = StrategySignal(
        symbol="BTCUSDT",
        timeframe="1h",
        horizon_bars=4,
        signal="BUY",
        confidence=0.20,
    )
    routed_exit = asyncio.run(
        service.route_signals(
            509,
            signals=(low_conf_signal,),
            prices={"BTCUSDT": 100.0},
            planner_config=PlannerConfig(
                total_risk_budget_frac=0.10,
                max_symbol_exposure_frac=0.10,
                min_confidence=0.55,
            ),
        )
    )
    assert len(routed_exit) == 1

    diagnostics = service.get_execution_diagnostics(509)
    assert diagnostics is not None
    assert diagnostics.total_orders == 3
    assert diagnostics.accepted_orders == 3
    assert diagnostics.entry_orders == 1
    assert diagnostics.rebalance_orders == 1
    assert diagnostics.exit_orders == 1


def test_routed_execution_service_deadband_skips_small_rebalance() -> None:
    service = RoutedExecutionService(
        min_rebalance_notional_usd=200.0,
        risk_policy=PortfolioRiskPolicy(
            max_symbol_exposure_frac=0.10,
            max_gross_exposure_frac=0.20,
            max_net_exposure_frac=0.20,
        ),
    )
    assert asyncio.run(service.start_session(SessionRequest(user_id=510, live=False))) is True

    buy_signal = StrategySignal(
        symbol="BTCUSDT",
        timeframe="1h",
        horizon_bars=4,
        signal="BUY",
        confidence=0.90,
    )

    routed_entry = asyncio.run(
        service.route_signals(
            510,
            signals=(buy_signal,),
            prices={"BTCUSDT": 100.0},
            planner_config=PlannerConfig(
                total_risk_budget_frac=0.10,
                max_symbol_exposure_frac=0.10,
                min_confidence=0.0,
            ),
        )
    )
    assert len(routed_entry) == 1
    assert routed_entry[0].accepted is True

    routed_rebalance = asyncio.run(
        service.route_signals(
            510,
            signals=(buy_signal,),
            prices={"BTCUSDT": 100.0},
            planner_config=PlannerConfig(
                total_risk_budget_frac=0.09,
                max_symbol_exposure_frac=0.10,
                min_confidence=0.0,
            ),
        )
    )
    assert len(routed_rebalance) == 1
    assert routed_rebalance[0].accepted is False
    assert routed_rebalance[0].reason.startswith("skipped_by_deadband")

    diagnostics = service.get_execution_diagnostics(510)
    assert diagnostics is not None
    assert diagnostics.total_orders == 1
    assert diagnostics.skipped_by_deadband == 1


def test_routed_execution_service_filter_skips_do_not_count_as_rejects() -> None:
    class FilterSkippingAdapter:
        def get_positions(self):
            return {}

        def place_order(self, plan, *, idempotency_key: str, mark_price: float | None = None):
            return ExecutionResult(
                accepted=False,
                order_id="",
                idempotency_key=idempotency_key,
                symbol=plan.symbol,
                side=plan.side,
                requested_qty=plan.quantity,
                filled_qty=0.0,
                avg_price=float(mark_price or 0.0),
                status="skipped",
                created_at=datetime.now(timezone.utc).isoformat(),
                reason="skipped_by_filter:min_notional",
            )

    service = RoutedExecutionService(paper_adapter_factory=FilterSkippingAdapter)
    assert asyncio.run(service.start_session(SessionRequest(user_id=511, live=False))) is True

    signal = StrategySignal(
        symbol="BTCUSDT",
        timeframe="1h",
        horizon_bars=4,
        signal="BUY",
        confidence=0.9,
    )
    routed = asyncio.run(
        service.route_signals(
            511,
            signals=(signal,),
            prices={"BTCUSDT": 100.0},
        )
    )
    assert len(routed) == 1
    assert routed[0].reason.startswith("skipped_by_filter")

    diagnostics = service.get_execution_diagnostics(511)
    assert diagnostics is not None
    assert diagnostics.total_orders == 0
    assert diagnostics.rejected_orders == 0
    assert diagnostics.skipped_by_filter == 1


def test_routed_execution_service_populates_unrealized_symbol_pnl_for_paper() -> None:
    service = RoutedExecutionService()
    assert asyncio.run(service.start_session(SessionRequest(user_id=512, live=False))) is True

    buy_signal = StrategySignal(
        symbol="BTCUSDT",
        timeframe="1h",
        horizon_bars=4,
        signal="BUY",
        confidence=0.9,
    )

    routed = asyncio.run(
        service.route_signals(
            512,
            signals=(buy_signal,),
            prices={"BTCUSDT": 100.0},
            planner_config=PlannerConfig(
                total_risk_budget_frac=0.10,
                max_symbol_exposure_frac=0.10,
                min_confidence=0.0,
            ),
        )
    )
    assert len(routed) == 1

    asyncio.run(
        service.route_signals(
            512,
            signals=(buy_signal,),
            prices={"BTCUSDT": 110.0},
            planner_config=PlannerConfig(
                total_risk_budget_frac=0.10,
                max_symbol_exposure_frac=0.10,
                min_confidence=0.0,
            ),
        )
    )

    snapshot = service.get_portfolio_snapshot(512)
    assert snapshot is not None
    assert "BTCUSDT" in snapshot.symbol_pnl_usd
    open_qty = float(snapshot.open_positions.get("BTCUSDT", 0.0))
    assert snapshot.symbol_pnl_usd["BTCUSDT"] == pytest.approx(open_qty * 10.0)


def test_routed_execution_service_merges_prices_for_symbol_notional_snapshot() -> None:
    service = RoutedExecutionService(
        risk_policy=PortfolioRiskPolicy(
            max_symbol_exposure_frac=0.10,
            max_gross_exposure_frac=0.20,
            max_net_exposure_frac=0.20,
        )
    )
    assert asyncio.run(service.start_session(SessionRequest(user_id=513, live=False))) is True

    btc_signal = StrategySignal(
        symbol="BTCUSDT",
        timeframe="1h",
        horizon_bars=4,
        signal="BUY",
        confidence=0.9,
    )
    eth_signal = StrategySignal(
        symbol="ETHUSDT",
        timeframe="1h",
        horizon_bars=4,
        signal="BUY",
        confidence=0.9,
    )

    asyncio.run(
        service.route_signals(
            513,
            signals=(btc_signal,),
            prices={"BTCUSDT": 100.0},
            planner_config=PlannerConfig(
                total_risk_budget_frac=0.10,
                max_symbol_exposure_frac=0.10,
                min_confidence=0.0,
            ),
        )
    )
    asyncio.run(
        service.route_signals(
            513,
            signals=(eth_signal,),
            prices={"ETHUSDT": 200.0},
            planner_config=PlannerConfig(
                total_risk_budget_frac=0.10,
                max_symbol_exposure_frac=0.10,
                min_confidence=0.0,
            ),
        )
    )

    snapshot = service.get_portfolio_snapshot(513)
    assert snapshot is not None
    assert "BTCUSDT" in snapshot.open_positions
    assert "ETHUSDT" in snapshot.open_positions
    assert "BTCUSDT" in snapshot.symbol_notional_usd
    assert "ETHUSDT" in snapshot.symbol_notional_usd


def test_routed_execution_service_snapshot_refreshes_mark_prices_from_live_metrics() -> None:
    class DynamicLiveAdapter:
        def __init__(self) -> None:
            self._positions = {"BTCUSDT": 2.0}
            self._metric_calls = 0

        def get_positions(self):
            return dict(self._positions)

        def get_position_metrics(self):
            self._metric_calls += 1
            mark_price = 100.0 if self._metric_calls == 1 else 120.0
            qty = float(self._positions["BTCUSDT"])
            entry_price = 90.0
            return {
                "BTCUSDT": {
                    "entry_price": entry_price,
                    "unrealized_pnl_usd": (mark_price - entry_price) * qty,
                }
            }

    adapter = DynamicLiveAdapter()
    service = RoutedExecutionService(
        live_adapter_factory=lambda request: adapter,
        allow_live_execution=True,
        risk_policy=PortfolioRiskPolicy(
            max_symbol_exposure_frac=1.0,
            max_gross_exposure_frac=1.0,
            max_net_exposure_frac=1.0,
        ),
    )
    req = SessionRequest(
        user_id=5131,
        live=True,
        credentials={"binance_api_key": "k", "binance_api_secret": "s"},
    )
    assert asyncio.run(service.start_session(req)) is True

    first_snapshot = service.get_portfolio_snapshot(5131)
    assert first_snapshot is not None
    assert first_snapshot.symbol_notional_usd["BTCUSDT"] == pytest.approx(200.0)

    second_snapshot = service.get_portfolio_snapshot(5131)
    assert second_snapshot is not None
    assert second_snapshot.symbol_notional_usd["BTCUSDT"] == pytest.approx(240.0)
    assert service.get_last_prices(5131)["BTCUSDT"] == pytest.approx(120.0)


def test_routed_execution_service_enforces_aggregate_caps_across_sequential_symbol_callbacks() -> None:
    service = RoutedExecutionService(
        risk_policy=PortfolioRiskPolicy(
            max_symbol_exposure_frac=0.05,
            max_gross_exposure_frac=0.20,
            max_net_exposure_frac=0.10,
        )
    )
    assert asyncio.run(service.start_session(SessionRequest(user_id=514, live=False))) is True

    planner_cfg = PlannerConfig(
        total_risk_budget_frac=0.15,
        max_symbol_exposure_frac=0.05,
        min_confidence=0.0,
    )
    for symbol, price in (
        ("BTCUSDT", 100.0),
        ("ETHUSDT", 200.0),
        ("SOLUSDT", 50.0),
    ):
        signal = StrategySignal(
            symbol=symbol,
            timeframe="1h",
            horizon_bars=4,
            signal="SELL",
            confidence=0.9,
        )
        asyncio.run(
            service.route_signals(
                514,
                signals=(signal,),
                prices={symbol: price},
                planner_config=planner_cfg,
            )
        )

    snapshot = service.get_portfolio_snapshot(514)
    assert snapshot is not None
    assert snapshot.risk is not None
    assert snapshot.risk.gross_exposure_frac == pytest.approx(0.10, rel=0.03)
    assert snapshot.risk.net_exposure_frac == pytest.approx(-0.10, rel=0.03)


def test_routed_execution_service_sync_positions_restores_and_flattens_snapshot() -> None:
    service = RoutedExecutionService()
    assert asyncio.run(service.start_session(SessionRequest(user_id=515, live=False))) is True

    restored = asyncio.run(
        service.sync_positions(
            515,
            target_positions={"BTCUSDT": 2.5, "ETHUSDT": -3.0},
            prices={"BTCUSDT": 100.0, "ETHUSDT": 200.0},
        )
    )
    assert len(restored) == 2
    assert all(item.accepted for item in restored)

    snapshot = service.get_portfolio_snapshot(515)
    assert snapshot is not None
    assert float(snapshot.open_positions.get("BTCUSDT", 0.0)) == pytest.approx(2.5)
    assert float(snapshot.open_positions.get("ETHUSDT", 0.0)) == pytest.approx(-3.0)

    flattened = asyncio.run(
        service.sync_positions(
            515,
            target_positions={},
            prices={"BTCUSDT": 100.0, "ETHUSDT": 200.0},
        )
    )
    assert len(flattened) == 2
    assert all(item.accepted for item in flattened)

    flat_snapshot = service.get_portfolio_snapshot(515)
    assert flat_snapshot is not None
    assert flat_snapshot.open_positions == {}


def test_routed_execution_service_reset_session_state_reinitializes_paper_session() -> None:
    service = RoutedExecutionService()
    assert asyncio.run(service.start_session(SessionRequest(user_id=409, live=False))) is True

    signal = StrategySignal(
        symbol="BTCUSDT",
        timeframe="1h",
        horizon_bars=4,
        signal="BUY",
        confidence=0.8,
    )
    prices = {"BTCUSDT": 50000.0}

    routed_before = asyncio.run(service.route_signals(409, signals=(signal,), prices=prices))
    assert len(routed_before) == 1

    diagnostics_before = service.get_execution_diagnostics(409)
    assert diagnostics_before is not None
    assert diagnostics_before.total_orders == 1

    paused_eval = service.set_monitoring_snapshot(
        409,
        MonitoringSnapshot(feature_drift_alert=True),
    )
    assert paused_eval.pause_trading is True

    assert service.reset_session_state(409) is True
    assert service.is_running(409) is True
    assert service.get_session_mode(409) == "paper"

    diagnostics_after = service.get_execution_diagnostics(409)
    assert diagnostics_after is not None
    assert diagnostics_after.total_orders == 0
    assert diagnostics_after.accepted_orders == 0
    assert diagnostics_after.rejected_orders == 0

    evaluation = service.get_kill_switch_evaluation(409)
    assert evaluation is not None
    assert evaluation.pause_trading is False

    snap = service.get_portfolio_snapshot(409)
    assert snap is not None
    assert snap.open_positions == {}

    routed_after = asyncio.run(service.route_signals(409, signals=(signal,), prices=prices))
    assert len(routed_after) == 1


def test_routed_execution_service_clear_execution_diagnostics_keeps_session_running() -> None:
    service = RoutedExecutionService()
    assert asyncio.run(service.start_session(SessionRequest(user_id=4091, live=False))) is True

    signal = StrategySignal(
        symbol="BTCUSDT",
        timeframe="1h",
        horizon_bars=4,
        signal="BUY",
        confidence=0.8,
    )
    prices = {"BTCUSDT": 50000.0}

    routed_before = asyncio.run(service.route_signals(4091, signals=(signal,), prices=prices))
    assert len(routed_before) == 1

    diagnostics_before = service.get_execution_diagnostics(4091)
    assert diagnostics_before is not None
    assert diagnostics_before.total_orders == 1

    assert service.clear_execution_diagnostics(4091) is True
    assert service.is_running(4091) is True

    diagnostics_after = service.get_execution_diagnostics(4091)
    assert diagnostics_after is not None
    assert diagnostics_after.total_orders == 0
    assert diagnostics_after.accepted_orders == 0
    assert diagnostics_after.rejected_orders == 0

    routed_after = asyncio.run(service.route_signals(4091, signals=(signal,), prices=prices))
    assert len(routed_after) == 0


def test_routed_execution_service_reset_session_state_rejects_live_and_missing() -> None:
    class FakeAdapter:
        def get_positions(self):
            return {}

    service = RoutedExecutionService(
        paper_adapter_factory=FakeAdapter,
        live_adapter_factory=lambda request: FakeAdapter(),
    )
    assert service.reset_session_state(997) is False

    req = SessionRequest(
        user_id=410,
        live=True,
        credentials={"binance_api_key": "k", "binance_api_secret": "s"},
    )
    assert asyncio.run(service.start_session(req)) is True
    assert service.get_session_mode(410) == "live"
    assert service.reset_session_state(410) is False


def test_routed_execution_service_applies_canary_risk_cap_for_live_sessions() -> None:
    class CapturingLiveAdapter:
        def __init__(self) -> None:
            self._positions: dict[str, float] = {}
            self._seq = 0

        def get_positions(self):
            return dict(self._positions)

        def place_order(self, plan, *, idempotency_key: str, mark_price: float | None = None):
            self._seq += 1
            direction = 1.0 if plan.side == "BUY" else -1.0
            self._positions[plan.symbol] = self._positions.get(plan.symbol, 0.0) + (direction * plan.quantity)
            return ExecutionResult(
                accepted=True,
                order_id=f"live-{self._seq}",
                idempotency_key=idempotency_key,
                symbol=plan.symbol,
                side=plan.side,
                requested_qty=plan.quantity,
                filled_qty=plan.quantity,
                avg_price=float(mark_price or 0.0),
                status="filled",
                created_at=datetime.now(timezone.utc).isoformat(),
            )

    service = RoutedExecutionService(
        live_adapter_factory=lambda request: CapturingLiveAdapter(),
        allow_live_execution=True,
        canary_live_risk_cap_frac=0.10,
        planner_config=PlannerConfig(
            total_risk_budget_frac=0.90,
            max_symbol_exposure_frac=0.90,
            min_confidence=0.0,
        ),
        risk_policy=PortfolioRiskPolicy(
            max_symbol_exposure_frac=0.90,
            max_gross_exposure_frac=0.90,
            max_net_exposure_frac=0.90,
        ),
    )

    req = SessionRequest(
        user_id=411,
        live=True,
        credentials={"binance_api_key": "k", "binance_api_secret": "s"},
    )
    assert asyncio.run(service.start_session(req)) is True
    assert service.get_session_mode(411) == "live"

    signal = StrategySignal(
        symbol="BTCUSDT",
        timeframe="1h",
        horizon_bars=4,
        signal="BUY",
        confidence=0.95,
    )
    routed = asyncio.run(
        service.route_signals(
            411,
            signals=(signal,),
            prices={"BTCUSDT": 100.0},
        )
    )
    assert len(routed) == 1
    assert routed[0].requested_qty == pytest.approx(10.0)

    snapshot = service.get_portfolio_snapshot(411)
    assert snapshot is not None
    assert snapshot.risk is not None
    assert snapshot.risk.gross_exposure_frac == pytest.approx(0.10)
    assert snapshot.risk.risk_budget_used_frac == pytest.approx(1.0)


def test_routed_execution_service_blocks_live_start_when_go_no_go_fails() -> None:
    class FakeLiveAdapter:
        def get_positions(self):
            return {}

    service = RoutedExecutionService(
        live_adapter_factory=lambda request: FakeLiveAdapter(),
        allow_live_execution=True,
        enforce_live_go_no_go=True,
        live_go_no_go=False,
    )

    req = SessionRequest(
        user_id=412,
        live=True,
        credentials={"binance_api_key": "k", "binance_api_secret": "s"},
    )
    with pytest.raises(RuntimeError, match="go_no_go_failed"):
        asyncio.run(service.start_session(req))
    assert service.is_running(412) is False


def test_routed_execution_service_trips_rollback_gate_after_consecutive_live_failures() -> None:
    class CountingLiveAdapter:
        def __init__(self) -> None:
            self.place_calls = 0
            self._positions: dict[str, float] = {}

        def get_positions(self):
            return dict(self._positions)

        def place_order(self, plan, *, idempotency_key: str, mark_price: float | None = None):
            self.place_calls += 1
            side_sign = 1.0 if plan.side == "BUY" else -1.0
            self._positions[plan.symbol] = self._positions.get(plan.symbol, 0.0) + (side_sign * plan.quantity)
            return ExecutionResult(
                accepted=True,
                order_id=f"fill-{self.place_calls}",
                idempotency_key=idempotency_key,
                symbol=plan.symbol,
                side=plan.side,
                requested_qty=plan.quantity,
                filled_qty=plan.quantity,
                avg_price=float(mark_price or 0.0),
                status="filled",
                created_at=datetime.now(timezone.utc).isoformat(),
            )

    adapter = CountingLiveAdapter()
    service = RoutedExecutionService(
        live_adapter_factory=lambda request: adapter,
        allow_live_execution=True,
        enforce_live_go_no_go=True,
        live_go_no_go=True,
        rollback_failure_threshold=2,
    )
    req = SessionRequest(
        user_id=413,
        live=True,
        credentials={"binance_api_key": "k", "binance_api_secret": "s"},
    )
    assert asyncio.run(service.start_session(req)) is True
    assert service.get_session_mode(413) == "live"

    signal = StrategySignal(
        symbol="BTCUSDT",
        timeframe="1h",
        horizon_bars=4,
        signal="BUY",
        confidence=0.9,
    )

    first_blocked = asyncio.run(
        service.route_signals(
            413,
            signals=(signal,),
            prices={"BTCUSDT": 50000.0},
            monitoring_snapshot=MonitoringSnapshot(hard_risk_breach=True),
        )
    )
    assert first_blocked == ()

    second_blocked = asyncio.run(
        service.route_signals(
            413,
            signals=(signal,),
            prices={"BTCUSDT": 50000.0},
            monitoring_snapshot=MonitoringSnapshot(hard_risk_breach=True),
        )
    )
    assert second_blocked == ()

    blocked_after_rollback = asyncio.run(
        service.route_signals(
            413,
            signals=(signal,),
            prices={"BTCUSDT": 50000.0},
            monitoring_snapshot=MonitoringSnapshot(),
        )
    )
    assert blocked_after_rollback == ()

    evaluation = service.get_kill_switch_evaluation(413)
    assert evaluation is not None
    assert evaluation.pause_trading is True
    assert "rollback_required" in evaluation.reasons

    diagnostics = service.get_execution_diagnostics(413)
    assert diagnostics is not None
    assert diagnostics.paused_cycles == 3
    assert diagnostics.blocked_actionable_signals == 3
    assert diagnostics.rollback_required is True
    assert diagnostics.rollout_failure_streak >= 2
    assert "rollback_required" in diagnostics.rollout_gate_reasons
    assert adapter.place_calls == 0
