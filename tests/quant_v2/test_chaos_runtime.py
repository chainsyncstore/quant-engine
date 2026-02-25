from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from quant_v2.contracts import StrategySignal
from quant_v2.execution.adapters import ExecutionResult
from quant_v2.execution.service import RoutedExecutionService, SessionRequest


def _buy_signal() -> StrategySignal:
    return StrategySignal(
        symbol="BTCUSDT",
        timeframe="1h",
        horizon_bars=4,
        signal="BUY",
        confidence=0.9,
    )


def test_chaos_adapter_api_failure_is_captured_as_rejected_result() -> None:
    class FailingAdapter:
        def get_positions(self):
            return {}

        def place_order(self, plan, *, idempotency_key: str, mark_price: float | None = None, limit_price: float | None = None, post_only: bool = False):
            raise RuntimeError("exchange_unreachable")

    service = RoutedExecutionService(paper_adapter_factory=FailingAdapter)
    assert asyncio.run(service.start_session(SessionRequest(user_id=501, live=False))) is True

    routed = asyncio.run(
        service.route_signals(
            501,
            signals=(_buy_signal(),),
            prices={"BTCUSDT": 50000.0},
        )
    )

    assert len(routed) == 1
    assert routed[0].accepted is False
    assert routed[0].status == "error"
    assert routed[0].reason.startswith("adapter_exception:")

    diagnostics = service.get_execution_diagnostics(501)
    assert diagnostics is not None
    assert diagnostics.rejected_orders == 1

    # Kill-switch should pause next cycle from execution anomaly.
    blocked = asyncio.run(
        service.route_signals(
            501,
            signals=(_buy_signal(),),
            prices={"BTCUSDT": 50000.0},
        )
    )
    assert blocked == ()


def test_chaos_delayed_fill_status_does_not_crash_routing() -> None:
    class DelayedFillAdapter:
        def __init__(self) -> None:
            self._positions: dict[str, float] = {}

        def get_positions(self):
            return dict(self._positions)

        def place_order(self, plan, *, idempotency_key: str, mark_price: float | None = None, limit_price: float | None = None, post_only: bool = False):
            return ExecutionResult(
                accepted=True,
                order_id="pending-1",
                idempotency_key=idempotency_key,
                symbol=plan.symbol,
                side=plan.side,
                requested_qty=plan.quantity,
                filled_qty=0.0,
                avg_price=float(mark_price or 0.0),
                status="open",
                created_at=datetime.now(timezone.utc).isoformat(),
                reason="pending_fill",
            )

    service = RoutedExecutionService(paper_adapter_factory=DelayedFillAdapter)
    assert asyncio.run(service.start_session(SessionRequest(user_id=502, live=False))) is True

    routed = asyncio.run(
        service.route_signals(
            502,
            signals=(_buy_signal(),),
            prices={"BTCUSDT": 50000.0},
        )
    )

    assert len(routed) == 1
    assert routed[0].status == "open"
    diagnostics = service.get_execution_diagnostics(502)
    assert diagnostics is not None
    assert diagnostics.total_orders == 1


def test_chaos_stale_price_data_skips_order_placement() -> None:
    class CountingAdapter:
        def __init__(self) -> None:
            self.calls = 0

        def get_positions(self):
            return {}

        def place_order(self, plan, *, idempotency_key: str, mark_price: float | None = None, limit_price: float | None = None, post_only: bool = False):
            self.calls += 1
            return ExecutionResult(
                accepted=True,
                order_id="ok",
                idempotency_key=idempotency_key,
                symbol=plan.symbol,
                side=plan.side,
                requested_qty=plan.quantity,
                filled_qty=plan.quantity,
                avg_price=float(mark_price or 0.0),
                status="filled",
                created_at=datetime.now(timezone.utc).isoformat(),
            )

    adapter = CountingAdapter()
    service = RoutedExecutionService(paper_adapter_factory=lambda: adapter)
    assert asyncio.run(service.start_session(SessionRequest(user_id=503, live=False))) is True

    routed = asyncio.run(
        service.route_signals(
            503,
            signals=(_buy_signal(),),
            prices={"BTCUSDT": 0.0},
        )
    )

    assert routed == ()
    assert adapter.calls == 0


def test_chaos_restart_recovery_after_failure() -> None:
    state = {"fail_mode": True}

    class RecoveringAdapter:
        def __init__(self) -> None:
            self._positions: dict[str, float] = {}

        def get_positions(self):
            return dict(self._positions)

        def place_order(self, plan, *, idempotency_key: str, mark_price: float | None = None, limit_price: float | None = None, post_only: bool = False):
            if state["fail_mode"]:
                raise RuntimeError("temporary_outage")
            self._positions[plan.symbol] = self._positions.get(plan.symbol, 0.0) + plan.quantity
            return ExecutionResult(
                accepted=True,
                order_id="ok-1",
                idempotency_key=idempotency_key,
                symbol=plan.symbol,
                side=plan.side,
                requested_qty=plan.quantity,
                filled_qty=plan.quantity,
                avg_price=float(mark_price or 0.0),
                status="filled",
                created_at=datetime.now(timezone.utc).isoformat(),
            )

    service = RoutedExecutionService(paper_adapter_factory=RecoveringAdapter)

    req = SessionRequest(user_id=504, live=False)
    assert asyncio.run(service.start_session(req)) is True
    first = asyncio.run(service.route_signals(504, signals=(_buy_signal(),), prices={"BTCUSDT": 50000.0}))
    assert len(first) == 1
    assert first[0].accepted is False

    assert asyncio.run(service.stop_session(504)) is True

    state["fail_mode"] = False
    assert asyncio.run(service.start_session(req)) is True
    recovered = asyncio.run(
        service.route_signals(504, signals=(_buy_signal(),), prices={"BTCUSDT": 50000.0})
    )
    assert len(recovered) == 1
    assert recovered[0].accepted is True
