"""
Paper trading adapter that satisfies the ExecutionAdapter interface.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

from execution.cost_model import CostModel, CostSide
from state.position_state import PositionState, PositionSide

from execution_live.adapter import ExecutionAdapter
from execution_live.event_logger import ExecutionEventLogger
from execution_live.order_models import (
    AccountState,
    ExecutionIntent,
    ExecutionReport,
    IntentAction,
    OrderStatus,
    PositionSnapshot,
)
from execution_live.risk_checks import RiskCheck


class PaperExecutionAdapter(ExecutionAdapter):
    """
    Simple in-memory paper broker.

    - Fills market intents immediately at bar-open price plus configured costs
    - Enforces single position per symbol (mirrors current simulator constraints)
    - Applies optional risk checks prior to execution
    - Logs every execution event
    """

    def __init__(
        self,
        cost_model: CostModel,
        initial_equity: float,
        risk_checks: Optional[List[RiskCheck]] = None,
        event_logger: Optional[ExecutionEventLogger] = None,
    ):
        self._cost_model = cost_model
        self._initial_equity = initial_equity
        self._cash = initial_equity
        self._positions: Dict[str, PositionState] = {}
        self._last_prices: Dict[str, float] = {}
        self._order_seq = 0
        self._orders: Dict[str, ExecutionReport] = {}
        self._risk_checks = risk_checks or []
        self._logger = event_logger or ExecutionEventLogger()

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def get_account_state(self) -> AccountState:
        positions = []
        equity = self._cash

        for symbol, position_state in self._positions.items():
            if not position_state.has_position:
                continue

            mark_price = self._last_prices.get(symbol, position_state.position.entry_price)
            unrealized = position_state.get_unrealized_pnl(mark_price)
            equity += position_state.position.entry_capital + unrealized

            positions.append(
                PositionSnapshot(
                    symbol=symbol,
                    quantity=position_state.position.size,
                    average_price=position_state.position.entry_price,
                    side=position_state.position.side.value,
                    unrealized_pnl=unrealized,
                )
            )

        return AccountState(
            equity=equity,
            cash=self._cash,
            buying_power=self._cash,
            timestamp=datetime.now(timezone.utc),
            positions=positions,
        )

    def get_positions(self) -> List[PositionSnapshot]:
        return self.get_account_state().positions

    def place_order(self, intent: ExecutionIntent) -> ExecutionReport:
        order_id = self._next_order_id()

        ok, reason = self._run_risk_checks(intent)
        if not ok:
            return self._record_report(
                ExecutionReport(
                    order_id=order_id,
                    status=OrderStatus.REJECTED,
                    intent=intent,
                    message=reason,
                )
            )

        if intent.action == IntentAction.CLOSE:
            report = self._close_position(order_id, intent)
        else:
            target_side = PositionSide.LONG if intent.action == IntentAction.BUY else PositionSide.SHORT
            report = self._open_position(order_id, intent, target_side)

        return self._record_report(report)

    def cancel_order(self, order_id: str) -> ExecutionReport:
        report = self._orders.get(order_id)
        if report is None:
            report = ExecutionReport(
                order_id=order_id,
                status=OrderStatus.REJECTED,
                intent=self._build_placeholder_intent(),
                message="Order not found",
            )
        else:
            # Immediate fill broker – nothing to cancel, echo status
            report = ExecutionReport(
                order_id=order_id,
                status=OrderStatus.CANCELLED,
                intent=report.intent,
                message="Order already processed; cancel is a no-op",
            )

        return self._record_report(report)

    # --------------------------------------------------------------------- #
    # Execution helpers
    # --------------------------------------------------------------------- #

    def _open_position(
        self,
        order_id: str,
        intent: ExecutionIntent,
        side: PositionSide,
    ) -> ExecutionReport:
        position_state = self._positions.setdefault(intent.symbol, PositionState())

        if position_state.has_position:
            return ExecutionReport(
                order_id=order_id,
                status=OrderStatus.REJECTED,
                intent=intent,
                message="Position already open. Close before opening another.",
            )

        base_price = self._determine_price(intent)
        cost_side = CostSide.BUY if side == PositionSide.LONG else CostSide.SELL
        effective_price = self._cost_model.apply_costs(base_price, cost_side)
        notional = effective_price * intent.quantity

        if notional > self._cash:
            return ExecutionReport(
                order_id=order_id,
                status=OrderStatus.REJECTED,
                intent=intent,
                message=f"Insufficient cash. Needed {notional:,.2f}, available {self._cash:,.2f}",
            )

        position_state.open_position(
            side=side,
            entry_price=effective_price,
            size=intent.quantity,
            entry_timestamp=intent.timestamp,
            entry_capital=notional,
        )

        self._cash -= notional
        self._last_prices[intent.symbol] = base_price

        return ExecutionReport(
            order_id=order_id,
            status=OrderStatus.FILLED,
            intent=intent,
            filled_quantity=intent.quantity,
            avg_fill_price=effective_price,
            cost_paid=self._cost_model.calculate_cost_amount(base_price, intent.quantity, cost_side),
        )

    def _close_position(self, order_id: str, intent: ExecutionIntent) -> ExecutionReport:
        position_state = self._positions.get(intent.symbol)
        if position_state is None or not position_state.has_position:
            return ExecutionReport(
                order_id=order_id,
                status=OrderStatus.REJECTED,
                intent=intent,
                message="No open position to close.",
            )

        position = position_state.position
        base_price = self._determine_price(intent, fallback=self._last_prices.get(intent.symbol))
        cost_side = CostSide.SELL if position.side == PositionSide.LONG else CostSide.BUY
        effective_price = self._cost_model.apply_costs(base_price, cost_side)

        realized_pnl = (
            (effective_price - position.entry_price) * position.size
            if position.side == PositionSide.LONG
            else (position.entry_price - effective_price) * position.size
        )

        self._cash = position.entry_capital + realized_pnl
        self._last_prices[intent.symbol] = base_price
        position_state.close_position()

        return ExecutionReport(
            order_id=order_id,
            status=OrderStatus.FILLED,
            intent=intent,
            filled_quantity=position.size,
            avg_fill_price=effective_price,
            realized_pnl=realized_pnl,
        )

    def _determine_price(self, intent: ExecutionIntent, fallback: Optional[float] = None) -> float:
        price = intent.reference_price or intent.limit_price or fallback
        if price is None:
            raise ValueError("ExecutionIntent must include reference_price or limit_price.")
        return price

    # --------------------------------------------------------------------- #
    # Utilities
    # --------------------------------------------------------------------- #

    def _next_order_id(self) -> str:
        self._order_seq += 1
        return f"PAPER-{self._order_seq}"

    def _run_risk_checks(self, intent: ExecutionIntent) -> tuple[bool, str]:
        account_state = self.get_account_state()
        for check in self._risk_checks:
            ok, reason = check.evaluate(intent, account_state)
            if not ok:
                payload = {
                    "check": check.__class__.__name__,
                    "reason": reason,
                    "intent": {
                        "symbol": intent.symbol,
                        "action": intent.action.value,
                        "quantity": intent.quantity,
                        "timestamp": intent.timestamp.isoformat(),
                    },
                }
                policy_label = getattr(check, "policy_label", None)
                if policy_label:
                    payload["policy_label"] = policy_label
                snapshot_fn = getattr(check, "policy_snapshot", None)
                if callable(snapshot_fn):
                    payload["policy"] = snapshot_fn()
                self._logger.log("risk_check_rejected", payload)
                return False, reason
        return True, "Approved"

    def _record_report(self, report: ExecutionReport) -> ExecutionReport:
        self._orders[report.order_id] = report
        self._logger.log(
            "execution_report",
            {
                "order_id": report.order_id,
                "status": report.status.value,
                "symbol": report.intent.symbol,
                "action": report.intent.action.value,
                "filled_quantity": report.filled_quantity,
                "avg_fill_price": report.avg_fill_price,
                "realized_pnl": report.realized_pnl,
                "message": report.message,
            },
        )
        return report

    def _build_placeholder_intent(self) -> ExecutionIntent:
        now = datetime.now(timezone.utc)
        # Use a tiny epsilon to satisfy the strict >0 constraint; report never executes.
        epsilon_quantity = 1e-6
        return ExecutionIntent(
            symbol="UNSPECIFIED",
            action=IntentAction.BUY,
            quantity=epsilon_quantity,
            timestamp=now,
            reference_price=0.0,
        )
