"""Reconcile target exposures with current positions into executable order deltas."""

from __future__ import annotations

from quant_v2.contracts import OrderPlan
from quant_v2.accounting import AccountingStore, LedgerProjection, LedgerReconciliationReport


def reconcile_target_exposures(
    target_exposure_frac: dict[str, float],
    *,
    current_positions_qty: dict[str, float],
    prices: dict[str, float],
    equity_usd: float,
    min_qty: float = 0.0,
) -> tuple[OrderPlan, ...]:
    """Compute order plans needed to move positions toward target exposures."""

    if equity_usd <= 0.0:
        raise ValueError("equity_usd must be positive")
    if min_qty < 0.0:
        raise ValueError("min_qty must be >= 0")

    symbols = sorted(set(target_exposure_frac) | set(current_positions_qty))
    plans: list[OrderPlan] = []

    for symbol in symbols:
        price = float(prices.get(symbol, 0.0))
        if price <= 0.0:
            continue

        target_frac = float(target_exposure_frac.get(symbol, 0.0))
        target_qty = (target_frac * equity_usd) / price
        current_qty = float(current_positions_qty.get(symbol, 0.0))
        delta_qty = target_qty - current_qty

        if abs(delta_qty) <= min_qty:
            continue

        side = "BUY" if delta_qty > 0 else "SELL"
        current_abs = abs(current_qty)
        target_abs = abs(target_qty)
        same_side_or_flat = abs(target_qty) <= 1e-12 or current_qty * target_qty > 0.0
        reduce_only = current_abs > 1e-12 and target_abs < current_abs and same_side_or_flat
        plans.append(
            OrderPlan(
                symbol=symbol,
                side=side,
                quantity=abs(delta_qty),
                reduce_only=reduce_only,
            )
        )

    return tuple(plans)


def reconcile_ledger_state(
    account_id: int,
    *,
    store: AccountingStore,
    adapter_positions: dict[str, float],
    checkpoint: LedgerProjection | None = None,
    symbol_tolerances: dict[str, float] | None = None,
    cash_tolerance: float = 0.01,
    equity_tolerance: float = 0.01,
    open_orders: dict[str, float] | None = None,
) -> LedgerReconciliationReport:
    """Compare ledger projection against external adapter state."""

    return store.reconcile(
        account_id,
        adapter_positions=adapter_positions,
        checkpoint=checkpoint,
        symbol_tolerances=symbol_tolerances,
        cash_tolerance=cash_tolerance,
        equity_tolerance=equity_tolerance,
        open_orders=open_orders,
    )
