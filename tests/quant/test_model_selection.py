from __future__ import annotations

from pathlib import Path

from quant.telebot.model_selection import find_latest_model, resolve_model_dir
from quant_v2.model_registry import ModelRegistry


def _write_model_config(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text('{"mode": "crypto"}', encoding="utf-8")


def test_resolve_model_dir_prefers_registry_active_version(tmp_path: Path) -> None:
    model_root = tmp_path / "models"
    fallback = model_root / "model_20260220_010000"
    _write_model_config(fallback)

    green_artifact = tmp_path / "green_model"
    _write_model_config(green_artifact)

    registry = ModelRegistry(tmp_path / "registry")
    registry.register_version("green", green_artifact)
    registry.set_active_version("green")

    resolved = resolve_model_dir(model_root, registry.root)

    assert resolved.model_dir == green_artifact.resolve()
    assert resolved.source == "registry_active"
    assert resolved.active_version_id == "green"
    assert resolved.warning == ""


def test_resolve_model_dir_falls_back_when_active_artifact_invalid(tmp_path: Path) -> None:
    model_root = tmp_path / "models"
    fallback = model_root / "model_20260220_020000"
    _write_model_config(fallback)

    broken_artifact = tmp_path / "broken_model"
    broken_artifact.mkdir(parents=True)

    registry = ModelRegistry(tmp_path / "registry")
    registry.register_version("broken", broken_artifact)
    registry.set_active_version("broken")

    resolved = resolve_model_dir(model_root, registry.root)

    assert resolved.model_dir == fallback.resolve()
    assert resolved.source == "latest"
    assert resolved.active_version_id == "broken"
    assert "falling back" in resolved.warning


def test_resolve_model_dir_returns_none_when_no_registry_or_fallback(tmp_path: Path) -> None:
    model_root = tmp_path / "models"
    registry = ModelRegistry(tmp_path / "registry")

    resolved = resolve_model_dir(model_root, registry.root)

    assert resolved.model_dir is None
    assert resolved.source == "none"
    assert resolved.active_version_id is None


def test_find_latest_model_prefers_root_config(tmp_path: Path) -> None:
    direct_root = tmp_path / "production"
    _write_model_config(direct_root)

    assert find_latest_model(direct_root) == direct_root.resolve()
