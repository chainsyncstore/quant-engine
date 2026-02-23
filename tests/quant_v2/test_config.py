from __future__ import annotations

from pathlib import Path

from quant_v2.config import default_universe_symbols, get_runtime_profile


def test_runtime_profile_defaults() -> None:
    get_runtime_profile.cache_clear()
    profile = get_runtime_profile()

    assert profile.universe.anchor_interval == "1h"
    assert profile.universe.context_intervals == ("4h",)
    assert profile.universe.phase2_symbol_cap == 14
    assert len(profile.universe.symbols) == 10
    assert "BTCUSDT" in profile.universe.symbols
    assert "ETHUSDT" in profile.universe.symbols


def test_registry_root_override(monkeypatch, tmp_path: Path) -> None:
    get_runtime_profile.cache_clear()

    override = tmp_path / "registry"
    monkeypatch.setenv("BOT_MODEL_REGISTRY_ROOT", str(override))

    profile = get_runtime_profile()
    assert profile.model_registry_root == override

    get_runtime_profile.cache_clear()


def test_default_universe_symbols() -> None:
    get_runtime_profile.cache_clear()
    symbols = default_universe_symbols()

    assert symbols[0] == "BTCUSDT"
    assert symbols[-1] == "LTCUSDT"
    assert len(symbols) == len(set(symbols))

    get_runtime_profile.cache_clear()
