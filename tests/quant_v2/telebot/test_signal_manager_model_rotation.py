"""Tests for V2SignalManager on_model_rotated hook.

Refs: audit_20260423 task P3-1
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from quant_v2.telebot.signal_manager import V2SignalManager


class _FakeClient:
    def __init__(self, bars: pd.DataFrame) -> None:
        self._bars = bars

    def fetch_historical(self, date_from, date_to, *, symbol: str, interval: str) -> pd.DataFrame:
        _ = (date_from, date_to, symbol, interval)
        return self._bars


def _sample_bars(*, n: int = 120) -> pd.DataFrame:
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    index = pd.date_range(end=end, periods=n, freq="h", tz="UTC")
    closes = [float(10_000 + i * 20) for i in range(len(index))]
    df = pd.DataFrame({"close": closes}, index=index)
    df["open"] = df["close"] * 0.99
    df["high"] = df["close"] * 1.01
    df["low"] = df["close"] * 0.98
    df["volume"] = 1000.0
    return df


def _make_manager(tmp_path: Path, hook: Any = None) -> V2SignalManager:
    bars = _sample_bars()
    return V2SignalManager(
        model_dir=tmp_path,
        symbols=("BTCUSDT",),
        loop_interval_seconds=3600,
        client_factory=lambda creds, live, symbol, interval: _FakeClient(bars),
        on_model_rotated=hook,
    )


@pytest.fixture
def sample_bars() -> pd.DataFrame:
    return _sample_bars()


def test_hook_fires_on_reload(tmp_path: Path) -> None:
    """Hook is called once when active model pointer changes."""
    hook = MagicMock()
    manager = _make_manager(tmp_path, hook=hook)

    fake_version_id = "model_20260421_192947"
    fake_pointer = MagicMock()
    fake_pointer.version_id = fake_version_id
    fake_pointer.artifact_dir = str(tmp_path)

    with patch.object(manager.registry, "get_active_version", return_value=fake_pointer):
        with patch.object(manager, "_resolve_active_model_path", return_value=tmp_path):
            with patch("quant_v2.models.trainer.load_model", return_value=MagicMock()):
                with patch.object(manager, "_build_featured_frame", return_value=None):
                    with patch("quant_v2.models.ensemble.HorizonEnsemble.from_directory", return_value=None):
                        # Run one cycle to trigger model reload
                        asyncio.run(_run_one_cycle(manager))

    hook.assert_called_once_with(fake_version_id, f"registry_active:{fake_version_id}")


def test_hook_not_called_when_model_unchanged(tmp_path: Path) -> None:
    """Hook should not fire when active model pointer has not changed."""
    hook = MagicMock()
    manager = _make_manager(tmp_path, hook=hook)

    fake_version_id = "model_20260421_192947"
    fake_pointer = MagicMock()
    fake_pointer.version_id = fake_version_id
    fake_pointer.artifact_dir = str(tmp_path)

    fake_model = MagicMock()
    fake_model._version_id = fake_version_id

    # Pre-populate active_model so version_id matches pointer
    manager.active_model = fake_model

    with patch.object(manager.registry, "get_active_version", return_value=fake_pointer):
        asyncio.run(_run_one_cycle(manager))

    hook.assert_not_called()


def test_hook_exception_does_not_propagate(tmp_path: Path, caplog: Any) -> None:
    """Hook failure is swallowed and logged; reload continues."""
    hook = MagicMock(side_effect=RuntimeError("hook broken"))
    manager = _make_manager(tmp_path, hook=hook)

    fake_version_id = "model_20260421_192947"
    fake_pointer = MagicMock()
    fake_pointer.version_id = fake_version_id
    fake_pointer.artifact_dir = str(tmp_path)

    with caplog.at_level(logging.WARNING, logger="quant_v2.telebot.signal_manager"):
        with patch.object(manager.registry, "get_active_version", return_value=fake_pointer):
            with patch.object(manager, "_resolve_active_model_path", return_value=tmp_path):
                with patch("quant_v2.models.trainer.load_model", return_value=MagicMock()):
                    with patch.object(manager, "_build_featured_frame", return_value=None):
                        with patch("quant_v2.models.ensemble.HorizonEnsemble.from_directory", return_value=None):
                            asyncio.run(_run_one_cycle(manager))

    hook.assert_called_once()
    assert "on_model_rotated hook failed" in caplog.text


async def _run_one_cycle(manager: V2SignalManager) -> None:
    """Run a single _run_cycle via a temporary session."""
    # Start a temporary session just to have a client available
    await manager.start_session(
        user_id=999,
        creds={"live": False},
        on_signal=lambda p: None,
        execute_orders=False,
    )
    session = manager.sessions[999]
    try:
        await manager._run_cycle(session)
    finally:
        await manager.stop_session(999)
