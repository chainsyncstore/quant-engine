from __future__ import annotations

from quant.telebot.main import _resolve_execution_backend


def test_resolve_execution_backend_defaults_to_v2_memory() -> None:
    assert _resolve_execution_backend("", allow_legacy_runtime=False) == "v2_memory"


def test_resolve_execution_backend_requires_explicit_legacy_opt_in() -> None:
    assert _resolve_execution_backend("v1", allow_legacy_runtime=False) == "v2_memory"
    assert _resolve_execution_backend("v1_legacy", allow_legacy_runtime=False) == "v2_memory"


def test_resolve_execution_backend_allows_legacy_when_enabled() -> None:
    assert _resolve_execution_backend("v1", allow_legacy_runtime=True) == "v1_legacy"
    assert _resolve_execution_backend("v1_legacy", allow_legacy_runtime=True) == "v1_legacy"


def test_resolve_execution_backend_preserves_supported_v2_modes() -> None:
    assert _resolve_execution_backend("v2", allow_legacy_runtime=False) == "v2"
    assert _resolve_execution_backend("v2_memory", allow_legacy_runtime=False) == "v2_memory"
    assert _resolve_execution_backend("v2_shadow_memory", allow_legacy_runtime=False) == "v2_shadow_memory"
