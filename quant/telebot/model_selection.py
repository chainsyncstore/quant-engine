"""Helpers for selecting active bot model artifacts with registry fallback."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from quant_v2.model_registry import ModelRegistry

_REQUIRED_HORIZON_FILES = ("model_2m.pkl", "model_4m.pkl", "model_8m.pkl")


@dataclass(frozen=True)
class ModelResolution:
    """Resolved model artifact location and source metadata."""

    model_dir: Path | None
    source: str
    active_version_id: str | None = None
    warning: str = ""


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def find_latest_model(
    root: Path = Path("models/production"),
    *,
    allow_legacy_fallback: bool | None = None,
) -> Path | None:
    """Find the latest production model directory under a root."""

    root = Path(root).expanduser()
    if allow_legacy_fallback is None:
        allow_legacy_fallback = _env_flag("BOT_MODEL_SELECTION_ALLOW_LATEST_FALLBACK", False)

    if root.is_dir() and _looks_like_model_artifact(root):
        return root.resolve()
    if not root.exists():
        return None

    subdirs = sorted(
        [x for x in root.iterdir() if x.is_dir() and "model_" in x.name],
        key=lambda x: x.name,
    )
    for candidate in reversed(subdirs):
        if _looks_like_model_artifact(candidate):
            return candidate.resolve()

    if allow_legacy_fallback and root.is_dir():
        # Explicit dev bootstrap: permit a direct root artifact even when no
        # registry pointer exists, but only if it passes the structural check.
        if _looks_like_model_artifact(root):
            return root.resolve()
    return None


def resolve_model_dir(model_root: Path | str, registry_root: Path | str) -> ModelResolution:
    """Resolve model path from active registry pointer without silent fallback."""

    root = Path(model_root).expanduser()
    registry = ModelRegistry(registry_root)
    active = registry.get_active_version()
    allow_latest_fallback = _env_flag("BOT_MODEL_SELECTION_ALLOW_LATEST_FALLBACK", False)

    if active is not None:
        active_dir = Path(active.artifact_dir).expanduser().resolve()
        if _looks_like_model_artifact(active_dir):
            return ModelResolution(
                model_dir=active_dir,
                source="registry_active",
                active_version_id=active.version_id,
            )

        warning = (
            "Active registry version points to missing/invalid artifact "
            f"{active_dir}; refusing to fall back to latest model discovery."
        )
        if allow_latest_fallback:
            latest = find_latest_model(root, allow_legacy_fallback=True)
            if latest is not None:
                return ModelResolution(
                    model_dir=latest,
                    source="latest",
                    active_version_id=active.version_id,
                    warning=warning + f" Explicit dev fallback resolved {latest.name}.",
                )
        return ModelResolution(
            model_dir=None,
            source="registry_invalid",
            active_version_id=active.version_id,
            warning=warning,
        )

    if allow_latest_fallback:
        latest = find_latest_model(root, allow_legacy_fallback=True)
        return ModelResolution(
            model_dir=latest,
            source="latest" if latest else "none",
        )
    return ModelResolution(
        model_dir=None,
        source="none",
    )


def _looks_like_model_artifact(path: Path) -> bool:
    """Minimal artifact validity check for runtime loading."""

    if not path.is_dir():
        return False
    # v1 legacy format uses config.json + .joblib files.
    if (path / "config.json").exists():
        return True
    # v2 multi-horizon format requires the full required horizon set plus
    # sidecar manifests because the runtime loader validates them strictly.
    required_models = [path / name for name in _REQUIRED_HORIZON_FILES]
    if all(model_path.exists() for model_path in required_models):
        if all(model_path.with_suffix(".manifest.json").exists() for model_path in required_models):
            return True
    return False
