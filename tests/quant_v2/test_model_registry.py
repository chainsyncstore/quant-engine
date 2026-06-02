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
    assert active.status == "active"


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


def test_register_version_defaults_to_candidate_status(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    artifact_dir = tmp_path / "candidate"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "model_4m.pkl").write_text("placeholder", encoding="utf-8")

    record = registry.register_version(
        "candidate_a",
        artifact_dir,
        metrics={"promotion_eligible": True},
    )

    assert record.status == "candidate"
    assert registry.get_active_version() is None
    assert registry.list_candidates()[0].version_id == "candidate_a"


def test_promote_version_updates_active_pointer_and_metadata(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    artifact_dir = tmp_path / "candidate"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "model_4m.pkl").write_text("placeholder", encoding="utf-8")

    registry.register_version(
        "candidate_a",
        artifact_dir,
        metrics={"promotion_eligible": True},
    )

    promoted = registry.promote_version(
        "candidate_a",
        promoted_by="test",
        notes="approved",
    )
    active = registry.get_active_version()

    assert active is not None
    assert active.version_id == "candidate_a"
    assert promoted.status == "active"
    assert promoted.promoted_by == "test"
    assert promoted.promoted_at
    assert promoted.promotion_notes == "approved"


def test_promote_version_rejects_unknown_or_ineligible_version(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    artifact_dir = tmp_path / "candidate"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "model_4m.pkl").write_text("placeholder", encoding="utf-8")

    registry.register_version(
        "candidate_bad",
        artifact_dir,
        metrics={"promotion_eligible": False},
    )

    with pytest.raises(ValueError):
        registry.promote_version("missing")

    with pytest.raises(ValueError):
        registry.promote_version("candidate_bad")


def test_mark_paper_quarantine_updates_candidate_status(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    artifact_dir = tmp_path / "candidate"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "model_4m.pkl").write_text("placeholder", encoding="utf-8")
    registry.register_version("candidate_a", artifact_dir)

    record = registry.mark_paper_quarantine("candidate_a", notes="paper run")

    assert record.status == "paper_quarantine"
    assert record.promotion_notes == "paper run"


def test_paper_quarantine_requires_paper_evaluation_before_promotion(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    artifact_dir = tmp_path / "candidate"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "model_4m.pkl").write_text("placeholder", encoding="utf-8")
    registry.register_version(
        "candidate_a",
        artifact_dir,
        metrics={"promotion_eligible": True, "paper_quarantine_required": True},
        status="paper_quarantine",
    )

    with pytest.raises(ValueError):
        registry.promote_version("candidate_a")

    evaluated = registry.record_paper_evaluation(
        "candidate_a",
        evaluation={"paper_trades": 25, "paper_pnl_usd": 42.0, "max_drawdown_frac": 0.01},
        promotion_eligible=True,
        notes="forward paper passed",
    )
    assert evaluated.status == "candidate"
    assert evaluated.metrics["paper_evaluation"]["promotion_eligible"] is True

    promoted = registry.promote_version("candidate_a", promoted_by="test")
    assert promoted.status == "active"
