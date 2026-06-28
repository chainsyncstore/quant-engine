from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_v2.research.trade_outcome_labels import (
    TradeOutcomeLabelConfig,
    build_trade_outcome_labels,
    build_trade_outcome_report,
)


def _frame(close: list[float], *, high: list[float] | None = None, low: list[float] | None = None, spread: list[float] | None = None, funding: list[float] | None = None) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=len(close), freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": close,
            "high": high or close,
            "low": low or close,
            "close": close,
        },
        index=timestamps,
    )
    frame["symbol"] = "BTCUSDT"
    if spread is not None:
        frame["spread_bps"] = spread
    if funding is not None:
        frame["funding_rate_bps"] = funding
    frame.index.name = "timestamp"
    return frame.reset_index().set_index(["timestamp", "symbol"]).sort_index()


def test_long_take_profit_label() -> None:
    frame = _frame([100.0, 100.0, 100.0], high=[100.0, 100.3, 100.1], low=[99.9, 99.9, 99.9])
    config = TradeOutcomeLabelConfig(horizon_bars=1, profit_target_bps=20.0, stop_loss_bps=30.0, dead_zone_bps=5.0, round_trip_cost_bps=0.0)

    labels = build_trade_outcome_labels(frame, config=config, side="long")

    assert labels.iloc[0] == pytest.approx(1.0)


def test_long_stop_loss_label() -> None:
    frame = _frame([100.0, 100.0, 100.0], high=[100.0, 100.1, 100.1], low=[99.9, 99.6, 99.9])
    config = TradeOutcomeLabelConfig(horizon_bars=1, profit_target_bps=20.0, stop_loss_bps=30.0, dead_zone_bps=5.0, round_trip_cost_bps=0.0)

    labels = build_trade_outcome_labels(frame, config=config, side="long")

    assert labels.iloc[0] == pytest.approx(0.0)


def test_same_bar_conflict_resolves_to_stop_by_default_and_can_prefer_take() -> None:
    frame = _frame([100.0, 100.0, 100.0], high=[100.0, 100.4, 100.0], low=[99.9, 99.6, 99.9])
    stop_cfg = TradeOutcomeLabelConfig(horizon_bars=1, profit_target_bps=20.0, stop_loss_bps=30.0, dead_zone_bps=5.0, round_trip_cost_bps=0.0)
    take_cfg = TradeOutcomeLabelConfig(
        horizon_bars=1,
        profit_target_bps=20.0,
        stop_loss_bps=30.0,
        dead_zone_bps=5.0,
        round_trip_cost_bps=0.0,
        prefer_stop_on_same_bar=False,
    )

    stop_labels = build_trade_outcome_labels(frame, config=stop_cfg, side="long")
    take_labels = build_trade_outcome_labels(frame, config=take_cfg, side="long")

    assert stop_labels.iloc[0] == pytest.approx(0.0)
    assert take_labels.iloc[0] == pytest.approx(1.0)


def test_short_take_profit_and_stop_loss_labels() -> None:
    tp_frame = _frame([100.0, 100.0, 100.0], high=[100.0, 100.1, 100.0], low=[99.9, 99.7, 99.9])
    sl_frame = _frame([100.0, 100.0, 100.0], high=[100.0, 100.4, 100.0], low=[99.9, 99.7, 99.9])
    config = TradeOutcomeLabelConfig(horizon_bars=1, profit_target_bps=20.0, stop_loss_bps=30.0, dead_zone_bps=5.0, round_trip_cost_bps=0.0)

    tp_labels = build_trade_outcome_labels(tp_frame, config=config, side="short")
    sl_labels = build_trade_outcome_labels(sl_frame, config=config, side="short")

    assert tp_labels.iloc[0] == pytest.approx(1.0)
    assert sl_labels.iloc[0] == pytest.approx(0.0)


def test_time_exit_profitable_and_unprofitable_labels() -> None:
    up_frame = _frame([100.0, 101.0, 102.0], high=[100.2, 101.2, 102.2], low=[99.8, 100.8, 101.8])
    down_frame = _frame([100.0, 99.0, 98.0], high=[100.1, 99.2, 98.2], low=[99.8, 98.8, 97.8])
    config = TradeOutcomeLabelConfig(horizon_bars=2, profit_target_bps=50.0, stop_loss_bps=60.0, dead_zone_bps=5.0, round_trip_cost_bps=0.0)

    up_labels = build_trade_outcome_labels(up_frame, config=config, side="long")
    down_labels = build_trade_outcome_labels(down_frame, config=config, side="long")

    assert up_labels.iloc[0] == pytest.approx(1.0)
    assert down_labels.iloc[0] == pytest.approx(0.0)


def test_insufficient_lookahead_returns_nan() -> None:
    frame = _frame([100.0, 101.0, 102.0], high=[100.1, 101.1, 102.1], low=[99.9, 100.9, 101.9])
    config = TradeOutcomeLabelConfig(horizon_bars=3, profit_target_bps=20.0, stop_loss_bps=30.0, dead_zone_bps=5.0, round_trip_cost_bps=0.0)

    labels = build_trade_outcome_labels(frame, config=config, side="long")

    assert np.isnan(labels.iloc[0])


def test_validation_errors_and_report_shape() -> None:
    frame = _frame([100.0, 100.0, 100.0], high=[100.0, 100.3, 100.1], low=[99.9, 99.9, 99.9], spread=[1.0, 1.0, 1.0], funding=[0.0, 0.0, 0.0])
    config = TradeOutcomeLabelConfig(horizon_bars=1, profit_target_bps=20.0, stop_loss_bps=30.0, dead_zone_bps=5.0, round_trip_cost_bps=0.0)

    report = build_trade_outcome_report(frame, config=config)

    assert report["policy_version"] == "trade_outcome_labels_v1"
    assert report["long"]["label_counts"]["take"] >= 1
    assert "BTCUSDT" in report["long"]["by_symbol"]
    assert "barrier_counts" in report["short"]

    with pytest.raises(ValueError, match="Missing required columns"):
        build_trade_outcome_labels(frame.drop(columns=["open"]), config=config, side="long")


def test_costs_can_flip_positive_gross_trade_into_skip() -> None:
    frame = _frame([100.0, 100.0, 100.0], high=[100.0, 100.3, 100.1], low=[99.9, 99.9, 99.9], spread=[0.0, 0.0, 0.0], funding=[0.0, 0.0, 0.0])
    config = TradeOutcomeLabelConfig(horizon_bars=1, profit_target_bps=20.0, stop_loss_bps=30.0, dead_zone_bps=5.0, round_trip_cost_bps=50.0)

    labels = build_trade_outcome_labels(frame, config=config, side="long")

    assert labels.iloc[0] == pytest.approx(0.0)
