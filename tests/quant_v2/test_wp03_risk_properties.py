from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from quant_v2.execution.reconciler import reconcile_target_exposures
from quant_v2.execution.service import RoutedExecutionService
from quant_v2.portfolio.risk_policy import HardRiskLimits, build_dynamic_operating_limits


_CAP = st.floats(min_value=0.02, max_value=1.0, allow_nan=False, allow_infinity=False)
_SIGMA = st.floats(min_value=0.01, max_value=50.0, allow_nan=False, allow_infinity=False)


@st.composite
def _hard_limits(draw) -> HardRiskLimits:
    gross = draw(st.floats(min_value=0.05, max_value=1.0, allow_nan=False, allow_infinity=False))
    net = draw(st.floats(min_value=0.01, max_value=gross, allow_nan=False, allow_infinity=False))
    symbol = draw(st.floats(min_value=0.01, max_value=min(gross, 0.5), allow_nan=False, allow_infinity=False))
    return HardRiskLimits(
        max_symbol_exposure_frac=symbol,
        max_gross_exposure_frac=gross,
        max_net_exposure_frac=net,
    )


@given(hard=_hard_limits(), sigma=_SIGMA)
def test_dynamic_caps_never_exceed_hard_caps(hard: HardRiskLimits, sigma: float) -> None:
    dynamic = build_dynamic_operating_limits(hard_limits=hard, sigma_60=sigma)
    assert dynamic is not None
    assert dynamic.max_symbol_exposure_frac <= hard.max_symbol_exposure_frac
    assert dynamic.max_gross_exposure_frac <= hard.max_gross_exposure_frac
    assert dynamic.max_net_exposure_frac <= hard.max_net_exposure_frac


@given(
    hard=_hard_limits(),
    sigma_low=st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
    sigma_step=st.floats(min_value=0.0001, max_value=10.0, allow_nan=False, allow_infinity=False),
)
def test_worsening_volatility_cannot_enlarge_limits(
    hard: HardRiskLimits,
    sigma_low: float,
    sigma_step: float,
) -> None:
    sigma_high = sigma_low + sigma_step
    low = build_dynamic_operating_limits(hard_limits=hard, sigma_60=sigma_low)
    high = build_dynamic_operating_limits(hard_limits=hard, sigma_60=sigma_high)
    assert low is not None and high is not None
    assert high.max_symbol_exposure_frac <= low.max_symbol_exposure_frac
    assert high.max_gross_exposure_frac <= low.max_gross_exposure_frac
    assert high.max_net_exposure_frac <= low.max_net_exposure_frac


@given(
    equity=st.floats(min_value=100.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False),
    price_a=st.floats(min_value=1.0, max_value=100_000.0, allow_nan=False, allow_infinity=False),
    price_b=st.floats(min_value=1.0, max_value=100_000.0, allow_nan=False, allow_infinity=False),
    current_a=st.floats(min_value=0.001, max_value=100.0, allow_nan=False, allow_infinity=False),
    current_b=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    reduction_ratio=st.floats(min_value=0.0, max_value=0.999, allow_nan=False, allow_infinity=False),
)
def test_risk_reducing_order_never_increases_symbol_or_gross_exposure(
    equity: float,
    price_a: float,
    price_b: float,
    current_a: float,
    current_b: float,
    reduction_ratio: float,
) -> None:
    current_positions = {
        "AAAUSDT": current_a,
        "BBBUSDT": current_b,
    }
    current_exposure_a = (current_a * price_a) / equity
    target_exposure_a = current_exposure_a * reduction_ratio
    target_exposure_b = (current_b * price_b) / equity

    plans = reconcile_target_exposures(
        {
            "AAAUSDT": target_exposure_a,
            "BBBUSDT": target_exposure_b,
        },
        current_positions_qty=current_positions,
        prices={"AAAUSDT": price_a, "BBBUSDT": price_b},
        equity_usd=equity,
    )
    by_symbol = {plan.symbol: plan for plan in plans}
    plan = by_symbol.get("AAAUSDT")
    assert plan is not None
    assert plan.reduce_only is True

    after_a = RoutedExecutionService._project_after_position(
        current_qty=current_a,
        side=plan.side,
        quantity=float(plan.quantity),
    )
    before_symbol = abs((current_a * price_a) / equity)
    after_symbol = abs((after_a * price_a) / equity)
    before_gross = before_symbol + abs((current_b * price_b) / equity)
    after_gross = after_symbol + abs((current_b * price_b) / equity)

    assert after_symbol <= before_symbol + 1e-9
    assert after_gross <= before_gross + 1e-9


def test_adverse_mark_buffer_blocks_trade_that_would_breach_under_mark_jump() -> None:
    hard = HardRiskLimits(
        max_symbol_exposure_frac=0.10,
        max_gross_exposure_frac=0.20,
        max_net_exposure_frac=0.20,
    )
    reason = RoutedExecutionService._projected_limit_breach_reason(
        current_positions={},
        prices={"BTCUSDT": 100.0},
        equity_usd=10_000.0,
        hard_policy=hard,
        adverse_mark_buffer_frac=0.02,
        symbol="BTCUSDT",
        side="BUY",
        quantity=9.9,
    )
    assert reason == "projected_symbol_cap"


def test_projected_limit_guard_rejects_non_positive_equity() -> None:
    hard = HardRiskLimits()
    reason = RoutedExecutionService._projected_limit_breach_reason(
        current_positions={},
        prices={"BTCUSDT": 100.0},
        equity_usd=0.0,
        hard_policy=hard,
        symbol="BTCUSDT",
        side="BUY",
        quantity=1.0,
    )
    assert reason == "non_positive_equity"
