"""Tests for SymbolScorecard prediction tracking and accuracy multiplier."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from quant_v2.telebot.symbol_scorecard import PredictionRecord, SymbolScorecard


def _now() -> datetime:
    return datetime.now(timezone.utc)


class TestRecordAndEvaluate:
    """Core record → evaluate → hit-rate flow."""

    def test_record_prediction_stores_entry(self) -> None:
        sc = SymbolScorecard(min_samples=1)
        sc.record_prediction("BTCUSDT", "BUY", 0.75, entry_price=100.0, horizon_bars=4)
        assert len(sc._predictions["BTCUSDT"]) == 1
        rec = sc._predictions["BTCUSDT"][0]
        assert rec.direction == "BUY"
        assert rec.confidence == 0.75
        assert rec.entry_price == 100.0
        assert not rec.resolved

    def test_ignores_hold_direction(self) -> None:
        sc = SymbolScorecard(min_samples=1)
        sc.record_prediction("BTCUSDT", "HOLD", 0.5, entry_price=100.0)
        assert "BTCUSDT" not in sc._predictions

    def test_ignores_zero_price(self) -> None:
        sc = SymbolScorecard(min_samples=1)
        sc.record_prediction("BTCUSDT", "BUY", 0.75, entry_price=0.0)
        assert "BTCUSDT" not in sc._predictions

    def test_evaluate_resolves_expired_predictions(self) -> None:
        sc = SymbolScorecard(min_samples=1)
        # Manually inject an expired prediction
        expired = PredictionRecord(
            symbol="BTCUSDT",
            direction="BUY",
            confidence=0.75,
            entry_price=100.0,
            predicted_at=_now() - timedelta(hours=6),
            horizon_expires=_now() - timedelta(hours=2),
        )
        sc._predictions["BTCUSDT"] = [expired]

        resolved = sc.evaluate_pending({"BTCUSDT": 105.0})
        assert resolved == 1
        assert expired.resolved is True
        assert expired.hit is True  # price went up, BUY was correct

    def test_evaluate_sell_hit(self) -> None:
        sc = SymbolScorecard(min_samples=1)
        expired = PredictionRecord(
            symbol="ETHUSDT",
            direction="SELL",
            confidence=0.80,
            entry_price=3000.0,
            predicted_at=_now() - timedelta(hours=6),
            horizon_expires=_now() - timedelta(hours=2),
        )
        sc._predictions["ETHUSDT"] = [expired]

        sc.evaluate_pending({"ETHUSDT": 2900.0})
        assert expired.hit is True  # price went down, SELL was correct

    def test_evaluate_buy_miss(self) -> None:
        sc = SymbolScorecard(min_samples=1)
        expired = PredictionRecord(
            symbol="BTCUSDT",
            direction="BUY",
            confidence=0.70,
            entry_price=100.0,
            predicted_at=_now() - timedelta(hours=6),
            horizon_expires=_now() - timedelta(hours=2),
        )
        sc._predictions["BTCUSDT"] = [expired]

        sc.evaluate_pending({"BTCUSDT": 95.0})
        assert expired.hit is False  # price went down, BUY was wrong

    def test_does_not_resolve_before_horizon(self) -> None:
        sc = SymbolScorecard(min_samples=1)
        pending = PredictionRecord(
            symbol="BTCUSDT",
            direction="BUY",
            confidence=0.75,
            entry_price=100.0,
            predicted_at=_now(),
            horizon_expires=_now() + timedelta(hours=2),
        )
        sc._predictions["BTCUSDT"] = [pending]

        resolved = sc.evaluate_pending({"BTCUSDT": 105.0})
        assert resolved == 0
        assert not pending.resolved

    def test_skips_symbols_without_price(self) -> None:
        sc = SymbolScorecard(min_samples=1)
        expired = PredictionRecord(
            symbol="BTCUSDT",
            direction="BUY",
            confidence=0.75,
            entry_price=100.0,
            predicted_at=_now() - timedelta(hours=6),
            horizon_expires=_now() - timedelta(hours=2),
        )
        sc._predictions["BTCUSDT"] = [expired]

        resolved = sc.evaluate_pending({"ETHUSDT": 3000.0})  # no BTC price
        assert resolved == 0
        assert not expired.resolved


class TestHitRateAndMultiplier:
    """Hit rate calculation and multiplier bands."""

    def _build_scorecard(self, hits: int, misses: int, symbol: str = "BTCUSDT") -> SymbolScorecard:
        """Build a scorecard with pre-resolved predictions."""
        sc = SymbolScorecard(min_samples=1, lookback_hours=72)
        records = []
        base_time = _now() - timedelta(hours=10)
        for i in range(hits):
            records.append(PredictionRecord(
                symbol=symbol, direction="BUY", confidence=0.75,
                entry_price=100.0,
                predicted_at=base_time + timedelta(hours=i * 0.1),
                horizon_expires=base_time + timedelta(hours=i * 0.1 + 4),
                resolved=True, hit=True,
            ))
        for i in range(misses):
            records.append(PredictionRecord(
                symbol=symbol, direction="BUY", confidence=0.70,
                entry_price=100.0,
                predicted_at=base_time + timedelta(hours=(hits + i) * 0.1),
                horizon_expires=base_time + timedelta(hours=(hits + i) * 0.1 + 4),
                resolved=True, hit=False,
            ))
        sc._predictions[symbol] = records
        return sc

    def test_hit_rate_returns_none_insufficient_data(self) -> None:
        sc = SymbolScorecard(min_samples=8)
        assert sc.get_hit_rate("BTCUSDT") is None

    def test_hit_rate_with_data(self) -> None:
        sc = self._build_scorecard(7, 3)  # 70% hit rate
        assert sc.get_hit_rate("BTCUSDT") == pytest.approx(0.7)

    def test_multiplier_strong_above_55(self) -> None:
        sc = self._build_scorecard(6, 4)  # 60% > 55%
        assert sc.get_accuracy_multiplier("BTCUSDT") == 1.0

    def test_multiplier_neutral_between_45_and_55(self) -> None:
        sc = self._build_scorecard(5, 5)  # 50% in [45%, 55%)
        assert sc.get_accuracy_multiplier("BTCUSDT") == 0.60

    def test_multiplier_weak_below_45(self) -> None:
        sc = self._build_scorecard(3, 7)  # 30% < 45%
        assert sc.get_accuracy_multiplier("BTCUSDT") == 0.30

    def test_multiplier_neutral_when_no_data(self) -> None:
        sc = SymbolScorecard(min_samples=8)
        assert sc.get_accuracy_multiplier("UNKNOWN") == 1.0


class TestSummaryAndReset:
    """Diagnostics and reset."""

    def test_get_summary_includes_all_symbols(self) -> None:
        sc = SymbolScorecard(min_samples=1)
        expired = PredictionRecord(
            symbol="BTCUSDT", direction="BUY", confidence=0.75,
            entry_price=100.0,
            predicted_at=_now() - timedelta(hours=6),
            horizon_expires=_now() - timedelta(hours=2),
            resolved=True, hit=True,
        )
        pending = PredictionRecord(
            symbol="BTCUSDT", direction="SELL", confidence=0.80,
            entry_price=105.0,
            predicted_at=_now(),
            horizon_expires=_now() + timedelta(hours=4),
        )
        sc._predictions["BTCUSDT"] = [expired, pending]

        summary = sc.get_summary()
        assert "BTCUSDT" in summary
        assert summary["BTCUSDT"]["resolved"] == 1
        assert summary["BTCUSDT"]["pending"] == 1
        assert summary["BTCUSDT"]["hits"] == 1
        assert summary["BTCUSDT"]["hit_rate"] == 1.0

    def test_reset_clears_all(self) -> None:
        sc = SymbolScorecard(min_samples=1)
        sc.record_prediction("BTCUSDT", "BUY", 0.75, entry_price=100.0)
        assert len(sc._predictions) == 1
        sc.reset()
        assert len(sc._predictions) == 0

    def test_prune_removes_old_resolved(self) -> None:
        sc = SymbolScorecard(min_samples=1, lookback_hours=24)
        old_resolved = PredictionRecord(
            symbol="BTCUSDT", direction="BUY", confidence=0.75,
            entry_price=100.0,
            predicted_at=_now() - timedelta(hours=48),
            horizon_expires=_now() - timedelta(hours=44),
            resolved=True, hit=True,
        )
        recent_resolved = PredictionRecord(
            symbol="BTCUSDT", direction="BUY", confidence=0.75,
            entry_price=100.0,
            predicted_at=_now() - timedelta(hours=2),
            horizon_expires=_now() - timedelta(hours=1),
            resolved=True, hit=False,
        )
        sc._predictions["BTCUSDT"] = [old_resolved, recent_resolved]
        sc._prune_old()
        assert len(sc._predictions["BTCUSDT"]) == 1
        assert sc._predictions["BTCUSDT"][0] is recent_resolved
