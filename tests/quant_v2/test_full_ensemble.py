"""Tests for FullEnsemble — combined LightGBM + Chronos predictions."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from quant_v2.models.ensemble import FullEnsemble


def _make_feature_row() -> pd.DataFrame:
    """Minimal single-row feature DataFrame."""
    return pd.DataFrame({"feat_a": [0.5], "feat_b": [0.3]})


def _make_close_series(n: int = 100, base: float = 100.0) -> pd.Series:
    rng = np.random.default_rng(42)
    prices = base * np.cumprod(1.0 + rng.normal(0.0002, 0.01, size=n))
    return pd.Series(prices, index=pd.date_range("2025-01-01", periods=n, freq="1h"))


class TestFullEnsemble:
    """Tests for FullEnsemble.predict()."""

    def test_both_sources_combined(self) -> None:
        """When both LightGBM and Chronos return, weighted combination is used."""
        mock_lgbm = MagicMock()
        mock_lgbm.predict.return_value = (0.70, 0.20)

        feature_row = _make_feature_row()
        close = _make_close_series()

        with patch(
            "quant_v2.models.chronos_wrapper.predict_next_bar_direction",
            return_value=(0.80, 0.15),
        ):
            ensemble = FullEnsemble(
                lgbm_ensemble=mock_lgbm,
                enable_chronos=True,
                lgbm_weight=0.65,
                chronos_weight=0.35,
            )
            prob, unc, agreement = ensemble.predict(feature_row, close)

        # Both agree on direction (>0.5 = up), so agreement = 1.0
        assert agreement == 1.0
        assert 0.0 <= prob <= 1.0
        assert 0.0 <= unc <= 1.0
        # Weighted: 0.65*0.70 + 0.35*0.80 = 0.455 + 0.28 = 0.735
        assert prob == pytest.approx(0.735, abs=0.01)

    def test_agreement_bonus_reduces_uncertainty(self) -> None:
        """When both models agree, uncertainty is reduced by 20%."""
        mock_lgbm = MagicMock()
        mock_lgbm.predict.return_value = (0.75, 0.40)

        feature_row = _make_feature_row()
        close = _make_close_series()

        with patch(
            "quant_v2.models.chronos_wrapper.predict_next_bar_direction",
            return_value=(0.80, 0.30),
        ):
            ensemble = FullEnsemble(lgbm_ensemble=mock_lgbm, enable_chronos=True)
            prob, unc, agreement = ensemble.predict(feature_row, close)

        assert agreement == 1.0
        # Without bonus: 0.65*0.40 + 0.35*0.30 = 0.365
        # With bonus: 0.365 * 0.80 = 0.292
        expected_unc = (0.65 * 0.40 + 0.35 * 0.30) * 0.80
        assert unc == pytest.approx(expected_unc, abs=0.01)

    def test_disagreement_no_bonus(self) -> None:
        """When models disagree, no uncertainty reduction and agreement = 0.0."""
        mock_lgbm = MagicMock()
        mock_lgbm.predict.return_value = (0.70, 0.30)  # bullish

        feature_row = _make_feature_row()
        close = _make_close_series()

        with patch(
            "quant_v2.models.chronos_wrapper.predict_next_bar_direction",
            return_value=(0.30, 0.25),  # bearish
        ):
            ensemble = FullEnsemble(lgbm_ensemble=mock_lgbm, enable_chronos=True)
            prob, unc, agreement = ensemble.predict(feature_row, close)

        assert agreement == 0.0
        # No bonus: raw weighted uncertainty
        expected_unc = 0.65 * 0.30 + 0.35 * 0.25
        assert unc == pytest.approx(expected_unc, abs=0.01)

    def test_fallback_when_chronos_fails(self) -> None:
        """When Chronos raises, FullEnsemble falls back to LightGBM only."""
        mock_lgbm = MagicMock()
        mock_lgbm.predict.return_value = (0.72, 0.25)

        feature_row = _make_feature_row()
        close = _make_close_series()

        with patch(
            "quant_v2.models.chronos_wrapper.predict_next_bar_direction",
            side_effect=ImportError("torch not installed"),
        ):
            ensemble = FullEnsemble(lgbm_ensemble=mock_lgbm, enable_chronos=True)
            prob, unc, agreement = ensemble.predict(feature_row, close)

        # Only one source → agreement is None
        assert agreement is None
        assert prob == pytest.approx(0.72)
        assert unc == pytest.approx(0.25)

    def test_fallback_when_lgbm_fails(self) -> None:
        """When LightGBM raises, FullEnsemble falls back to Chronos only."""
        mock_lgbm = MagicMock()
        mock_lgbm.predict.side_effect = RuntimeError("model corrupt")

        feature_row = _make_feature_row()
        close = _make_close_series()

        with patch(
            "quant_v2.models.chronos_wrapper.predict_next_bar_direction",
            return_value=(0.60, 0.35),
        ):
            ensemble = FullEnsemble(lgbm_ensemble=mock_lgbm, enable_chronos=True)
            prob, unc, agreement = ensemble.predict(feature_row, close)

        assert agreement is None
        assert prob == pytest.approx(0.60)
        assert unc == pytest.approx(0.35)

    def test_both_fail_returns_default(self) -> None:
        """When both sources fail, returns (0.5, 1.0, None)."""
        mock_lgbm = MagicMock()
        mock_lgbm.predict.side_effect = RuntimeError("boom")

        feature_row = _make_feature_row()
        close = _make_close_series()

        with patch(
            "quant_v2.models.chronos_wrapper.predict_next_bar_direction",
            side_effect=RuntimeError("also boom"),
        ):
            ensemble = FullEnsemble(lgbm_ensemble=mock_lgbm, enable_chronos=True)
            prob, unc, agreement = ensemble.predict(feature_row, close)

        assert prob == 0.5
        assert unc == 1.0
        assert agreement is None

    def test_chronos_disabled(self) -> None:
        """enable_chronos=False skips Chronos entirely."""
        mock_lgbm = MagicMock()
        mock_lgbm.predict.return_value = (0.68, 0.22)

        feature_row = _make_feature_row()
        close = _make_close_series()

        ensemble = FullEnsemble(lgbm_ensemble=mock_lgbm, enable_chronos=False)
        prob, unc, agreement = ensemble.predict(feature_row, close)

        # Only one source → agreement is None
        assert agreement is None
        assert prob == pytest.approx(0.68)
        assert unc == pytest.approx(0.22)

    def test_no_lgbm_chronos_only(self) -> None:
        """With no LightGBM ensemble, only Chronos contributes."""
        feature_row = _make_feature_row()
        close = _make_close_series()

        with patch(
            "quant_v2.models.chronos_wrapper.predict_next_bar_direction",
            return_value=(0.55, 0.40),
        ):
            ensemble = FullEnsemble(lgbm_ensemble=None, enable_chronos=True)
            prob, unc, agreement = ensemble.predict(feature_row, close)

        assert agreement is None
        assert prob == pytest.approx(0.55)
        assert unc == pytest.approx(0.40)
