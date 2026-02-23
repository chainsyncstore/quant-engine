"""Helpers for selecting active bot model artifacts with registry fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from quant_v2.model_registry import ModelRegistry


@dataclass(frozen=True)
class ModelResolution:
    """Resolved model artifact location and source metadata."""

    model_dir: Path | None
    source: str
    active_version_id: str | None = None
    warning: str = ""


def find_latest_model(root: Path = Path("models/production")) -> Path | None:
    """Find the latest production model directory under a root."""

    root = Path(root).expanduser()
    if root.is_dir() and (root / "config.json").exists():
        return root.resolve()
    if not root.exists():
        return None

    subdirs = sorted(
        [x for x in root.iterdir() if x.is_dir() and "model_" in x.name],
        key=lambda x: x.name,
    )
    if subdirs:
        return subdirs[-1].resolve()
    return None


def resolve_model_dir(model_root: Path | str, registry_root: Path | str) -> ModelResolution:
    """Resolve model path from active registry pointer, falling back to latest model."""

    root = Path(model_root).expanduser()
    registry = ModelRegistry(registry_root)
    active = registry.get_active_version()

    if active is not None:
        active_dir = Path(active.artifact_dir).expanduser().resolve()
        if _looks_like_model_artifact(active_dir):
            return ModelResolution(
                model_dir=active_dir,
                source="registry_active",
                active_version_id=active.version_id,
            )

        latest = find_latest_model(root)
        warning = (
            "Active registry version points to missing/invalid artifact "
            f"{active_dir}; falling back to latest model discovery."
        )
        return ModelResolution(
            model_dir=latest,
            source="latest" if latest else "none",
            active_version_id=active.version_id,
            warning=warning,
        )

    latest = find_latest_model(root)
    return ModelResolution(
        model_dir=latest,
        source="latest" if latest else "none",
    )


def _looks_like_model_artifact(path: Path) -> bool:
    """Minimal artifact validity check for runtime loading."""

    return path.is_dir() and (path / "config.json").exists()
