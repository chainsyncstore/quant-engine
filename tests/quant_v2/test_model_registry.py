from __future__ import annotations

from pathlib import Path

import pytest

from quant_v2.model_registry import ModelRegistry


def test_register_and_activate_version(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    artifact_dir = tmp_path / "model_artifacts"
    artifact_dir.mkdir(parents=True)

    registered = registry.register_version(
        "v2_20260223_0100",
        artifact_dir,
        metrics={"score": 73.2},
        tags={"mode": "shadow"},
    )

    assert registered.version_id == "v2_20260223_0100"
    assert Path(registered.artifact_dir) == artifact_dir.resolve()

    registry.set_active_version("v2_20260223_0100")
    active = registry.get_active_version()
    assert active is not None
    assert active.version_id == "v2_20260223_0100"
    assert active.metrics["score"] == 73.2


def test_set_active_version_rejects_unknown_version(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")

    with pytest.raises(ValueError):
        registry.set_active_version("missing_version")


def test_register_rejects_path_separator_version_id(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir(parents=True)

    with pytest.raises(ValueError):
        registry.register_version("bad/name", artifact_dir)


def test_active_pointer_tracks_previous_version_and_rollback(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")

    artifact_a = tmp_path / "artifact_a"
    artifact_a.mkdir(parents=True)
    artifact_b = tmp_path / "artifact_b"
    artifact_b.mkdir(parents=True)

    registry.register_version("v2_a", artifact_a)
    registry.register_version("v2_b", artifact_b)

    registry.set_active_version("v2_a")
    pointer_a = registry.get_active_pointer()
    assert pointer_a is not None
    assert pointer_a.version_id == "v2_a"
    assert pointer_a.previous_version_id is None

    registry.set_active_version("v2_b")
    pointer_b = registry.get_active_pointer()
    assert pointer_b is not None
    assert pointer_b.version_id == "v2_b"
    assert pointer_b.previous_version_id == "v2_a"

    previous = registry.get_previous_active_version()
    assert previous is not None
    assert previous.version_id == "v2_a"

    rolled_back = registry.rollback_to_previous_version()
    assert rolled_back is not None
    assert rolled_back.version_id == "v2_a"

    pointer_after = registry.get_active_pointer()
    assert pointer_after is not None
    assert pointer_after.version_id == "v2_a"
    assert pointer_after.previous_version_id == "v2_b"
