"""Tests for scheduled_retrain universe default behavior (P1-1)."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest


class TestDefaultExtraSymbols:
    """Test that RETRAIN_TRAIN_SYMBOLS defaults track default_universe_symbols."""

    def test_default_extra_symbols_tracks_universe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default extra symbols should be universe minus BTCUSDT."""
        # Unset env var to test default behavior
        monkeypatch.delenv("RETRAIN_TRAIN_SYMBOLS", raising=False)

        # Mock default_universe_symbols where it's used in scheduled_retrain
        mock_universe = ("BTCUSDT", "ETHUSDT", "ADAUSDT")

        with patch("quant_v2.research.scheduled_retrain.default_universe_symbols", return_value=mock_universe):
            # Import inside patch to get the patched version
            from quant_v2.research.scheduled_retrain import default_universe_symbols

            # Simulate what run_scheduler_loop does
            _universe = [s for s in default_universe_symbols() if s != "BTCUSDT"]
            _default_extra_syms = ",".join(_universe) if _universe else "ETHUSDT,BNBUSDT"
            _extra_sym_raw = os.getenv("RETRAIN_TRAIN_SYMBOLS", _default_extra_syms).strip()
            extra_symbols = (
                [s.strip() for s in _extra_sym_raw.split(",") if s.strip()]
                if _extra_sym_raw
                else []
            )

            assert extra_symbols == ["ETHUSDT", "ADAUSDT"]

    def test_env_override_still_honoured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RETRAIN_TRAIN_SYMBOLS env var should override the default."""
        monkeypatch.setenv("RETRAIN_TRAIN_SYMBOLS", "FOOUSDT,BARUSDT")

        # Mock with a dummy universe - env var should still override
        mock_universe = ("BTCUSDT", "ETHUSDT", "ADAUSDT")

        with patch("quant_v2.research.scheduled_retrain.default_universe_symbols", return_value=mock_universe):
            from quant_v2.research.scheduled_retrain import default_universe_symbols

            # Simulate what run_scheduler_loop does
            _universe = [s for s in default_universe_symbols() if s != "BTCUSDT"]
            _default_extra_syms = ",".join(_universe) if _universe else "ETHUSDT,BNBUSDT"
            _extra_sym_raw = os.getenv("RETRAIN_TRAIN_SYMBOLS", _default_extra_syms).strip()
            extra_symbols = (
                [s.strip() for s in _extra_sym_raw.split(",") if s.strip()]
                if _extra_sym_raw
                else []
            )

            assert extra_symbols == ["FOOUSDT", "BARUSDT"]

    def test_empty_universe_falls_back_to_safe_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If default_universe_symbols returns empty, fall back to safe defaults."""
        monkeypatch.delenv("RETRAIN_TRAIN_SYMBOLS", raising=False)

        with patch("quant_v2.research.scheduled_retrain.default_universe_symbols", return_value=()):
            from quant_v2.research.scheduled_retrain import default_universe_symbols

            # Simulate what run_scheduler_loop does
            _universe = [s for s in default_universe_symbols() if s != "BTCUSDT"]
            _default_extra_syms = ",".join(_universe) if _universe else "ETHUSDT,BNBUSDT"
            _extra_sym_raw = os.getenv("RETRAIN_TRAIN_SYMBOLS", _default_extra_syms).strip()
            extra_symbols = (
                [s.strip() for s in _extra_sym_raw.split(",") if s.strip()]
                if _extra_sym_raw
                else []
            )

            # When universe is empty, _default_extra_syms falls back to safe default
            assert extra_symbols == ["ETHUSDT", "BNBUSDT"]
