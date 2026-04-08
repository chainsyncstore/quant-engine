"""Tests for quant_v2.models.chronos_wrapper — all use mocks, no model download."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


class _FakeTensor:
    """Minimal tensor mock so tests do not require real torch."""

    def __init__(self, data: np.ndarray) -> None:
        self._data = np.asarray(data, dtype=np.float32)
        self.shape = self._data.shape

    def unsqueeze(self, dim: int) -> "_FakeTensor":
        return _FakeTensor(np.expand_dims(self._data, axis=dim))

    def numpy(self) -> np.ndarray:
        return self._data

    def __getitem__(self, idx):
        return _FakeTensor(self._data[idx])


def _build_fake_torch():
    """Return a minimal mock 'torch' module."""
    fake_torch = MagicMock()
    fake_torch.float32 = "float32"
    fake_torch.tensor = lambda data, dtype=None: _FakeTensor(np.asarray(data, dtype=np.float32))
    fake_torch.no_grad.return_value.__enter__ = lambda s: None
    fake_torch.no_grad.return_value.__exit__ = lambda s, *a: None
    return fake_torch


def _make_close_series(n: int = 256, base: float = 100.0) -> pd.Series:
    """Generate a synthetic close price series."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0002, 0.01, size=n)
    prices = base * np.cumprod(1.0 + returns)
    return pd.Series(prices, index=pd.date_range("2025-01-01", periods=n, freq="1h"))


class TestPredictNextBarDirection:
    """Tests for predict_next_bar_direction with mocked Chronos pipeline."""

    def test_returns_valid_prob_and_uncertainty(self) -> None:
        """Mocked pipeline returns a forecast; function returns valid tuple."""
        fake_torch = _build_fake_torch()

        mock_pipeline = MagicMock()
        current_price = 105.0
        samples = np.random.default_rng(7).normal(current_price * 1.01, 0.5, size=(50, 4))
        # pipeline.predict returns shape (1, num_samples, prediction_length)
        mock_pipeline.predict.return_value = _FakeTensor(samples).unsqueeze(0)

        close = _make_close_series(100, base=100.0)
        close.iloc[-1] = current_price

        with patch.dict(sys.modules, {"torch": fake_torch}):
            with patch("quant_v2.models.chronos_wrapper._get_pipeline", return_value=mock_pipeline):
                from quant_v2.models.chronos_wrapper import predict_next_bar_direction

                prob_up, uncertainty = predict_next_bar_direction(close, prediction_length=4)

        assert isinstance(prob_up, float)
        assert isinstance(uncertainty, float)
        assert 0.0 <= prob_up <= 1.0
        assert 0.0 <= uncertainty <= 1.0
        mock_pipeline.predict.assert_called_once()

    def test_all_samples_above_gives_high_prob(self) -> None:
        """When all forecast samples are above current price, prob_up ≈ 1.0."""
        fake_torch = _build_fake_torch()

        mock_pipeline = MagicMock()
        current_price = 100.0
        samples = np.full((50, 4), current_price * 1.05)
        mock_pipeline.predict.return_value = _FakeTensor(samples).unsqueeze(0)

        close = pd.Series(
            np.linspace(95.0, current_price, 100),
            index=pd.date_range("2025-01-01", periods=100, freq="1h"),
        )

        with patch.dict(sys.modules, {"torch": fake_torch}):
            with patch("quant_v2.models.chronos_wrapper._get_pipeline", return_value=mock_pipeline):
                from quant_v2.models.chronos_wrapper import predict_next_bar_direction

                prob_up, uncertainty = predict_next_bar_direction(close, prediction_length=4)

        assert prob_up == pytest.approx(1.0)

    def test_all_samples_below_gives_low_prob(self) -> None:
        """When all forecast samples are below current price, prob_up ≈ 0.0."""
        fake_torch = _build_fake_torch()

        mock_pipeline = MagicMock()
        current_price = 100.0
        samples = np.full((50, 4), current_price * 0.95)
        mock_pipeline.predict.return_value = _FakeTensor(samples).unsqueeze(0)

        close = pd.Series(
            np.linspace(95.0, current_price, 100),
            index=pd.date_range("2025-01-01", periods=100, freq="1h"),
        )

        with patch.dict(sys.modules, {"torch": fake_torch}):
            with patch("quant_v2.models.chronos_wrapper._get_pipeline", return_value=mock_pipeline):
                from quant_v2.models.chronos_wrapper import predict_next_bar_direction

                prob_up, uncertainty = predict_next_bar_direction(close, prediction_length=4)

        assert prob_up == pytest.approx(0.0)

    def test_insufficient_data_returns_default(self) -> None:
        """With fewer than 32 bars, returns (0.5, 1.0) — no torch needed."""
        close = pd.Series(
            np.linspace(90.0, 100.0, 20),
            index=pd.date_range("2025-01-01", periods=20, freq="1h"),
        )

        from quant_v2.models.chronos_wrapper import predict_next_bar_direction

        prob_up, uncertainty = predict_next_bar_direction(close, prediction_length=4)
        assert prob_up == 0.5
        assert uncertainty == 1.0

    def test_uses_last_256_bars_as_context(self) -> None:
        """Pipeline should receive at most 256 context values."""
        fake_torch = _build_fake_torch()

        mock_pipeline = MagicMock()
        samples = np.full((50, 4), 101.0)
        mock_pipeline.predict.return_value = _FakeTensor(samples).unsqueeze(0)

        close = _make_close_series(500)

        with patch.dict(sys.modules, {"torch": fake_torch}):
            with patch("quant_v2.models.chronos_wrapper._get_pipeline", return_value=mock_pipeline):
                from quant_v2.models.chronos_wrapper import predict_next_bar_direction

                predict_next_bar_direction(close, prediction_length=4)

        call_args = mock_pipeline.predict.call_args
        context_tensor = call_args[0][0]
        assert context_tensor.shape[1] == 256
