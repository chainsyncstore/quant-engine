from __future__ import annotations

import pandas as pd
import pytest

from quant_v2.strategy.regime import classify_latest


def _close(start: float, step: float, periods: int = 40) -> pd.Series:
    return pd.Series([start + step * idx for idx in range(periods)])


def test_regime1_momentum_risk_remains_normal() -> None:
    funding = pd.Series([0.0] * 40)

    state = classify_latest(
        _close(100.0, 2.0),
        funding,
        lookback=12,
        persistence_guard=1,
    )

    assert state.regime == 1
    assert state.regime_risk == pytest.approx(0.0)


def test_regime2_reversion_risk_is_more_conservative_than_regime1() -> None:
    funding = pd.Series([0.0] * 40)

    state = classify_latest(
        _close(200.0, -2.0),
        funding,
        lookback=12,
        persistence_guard=1,
    )

    assert state.regime == 2
    assert state.regime_risk == pytest.approx(0.35)


def test_regime3_and_4_risk_mapping_do_not_regress() -> None:
    neutral = classify_latest(
        pd.Series([100.0] * 40),
        pd.Series([0.0] * 40),
        lookback=12,
        persistence_guard=1,
    )
    adverse = classify_latest(
        _close(100.0, 2.0),
        pd.Series([3.0] * 40),
        lookback=12,
        persistence_guard=1,
    )

    assert neutral.regime == 3
    assert neutral.regime_risk == pytest.approx(0.5)
    assert adverse.regime == 4
    assert adverse.regime_risk == pytest.approx(1.0)
