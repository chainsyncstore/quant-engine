from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_v2.model_registry import ModelRegistry
from quant_v2.models.trainer import save_model_bundle, train


def _sample_training_data(rows: int = 180) -> tuple[pd.DataFrame, pd.Series]:
    idx = pd.date_range(end=datetime.now(timezone.utc), periods=rows, freq="h", tz="UTC")
    x1 = np.linspace(-2.0, 2.0, rows)
    x2 = np.sin(np.linspace(0.0, 8.0, rows))
    x3 = np.cos(np.linspace(0.0, 6.0, rows))
    logits = 0.8 * x1 + 0.5 * x2 - 0.3 * x3
    y = (logits > 0.0).astype(int)
    X = pd.DataFrame({"f1": x1, "f2": x2, "f3": x3}, index=idx)
    return X, pd.Series(y, index=idx)


def _write_valid_model_bundle(artifact_dir: Path, *, monkeypatch) -> Path:
    monkeypatch.setenv("QUANT_IMAGE", "registry.example/quant-bot@sha256:" + "a" * 64)
    X, y = _sample_training_data()
    trained = train(X, y, horizon=4, calibration_frac=0.2)
    artifact = artifact_dir / "model_4m.pkl"
    save_model_bundle(
        trained,
        artifact,
        metadata={
            "threshold": 0.60,
            "threshold_policy": {
                "source": "oof_dev_predictions",
                "selected_threshold": 0.60,
                "selected_accuracy": 0.63,
            },
        },
    )
    return artifact


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


def test_active_pointer_tracks_previous_version_and_rollback(tmp_path: Path, monkeypatch) -> None:
    registry = ModelRegistry(tmp_path / "registry")

    artifact_a = tmp_path / "artifact_a"
    artifact_a.mkdir(parents=True)
    artifact_b = tmp_path / "artifact_b"
    artifact_b.mkdir(parents=True)

    X, y = _sample_training_data()
    monkeypatch.setenv("QUANT_IMAGE", "registry.example/quant-bot@sha256:" + "a" * 64)
    trained_a = train(X, y, horizon=4, calibration_frac=0.2)
    save_model_bundle(trained_a, artifact_a / "model_4m.pkl")
    trained_b = train(X, y, horizon=4, calibration_frac=0.2)
    save_model_bundle(trained_b, artifact_b / "model_4m.pkl")

    registry.register_version("v2_a", artifact_a, metrics={"promotion_eligible": True})
    registry.register_version("v2_b", artifact_b, metrics={"promotion_eligible": True})

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
    event_types = [event.event_type for event in registry.list_registry_events()]
    assert event_types.count("version_registered") == 2
    assert event_types.count("active_pointer_set") >= 2
    registry.active_file.unlink()
    pointer_from_events = registry.get_active_pointer()
    assert pointer_from_events is not None
    assert pointer_from_events.version_id == "v2_b"
    assert pointer_from_events.previous_version_id == "v2_a"

    previous = registry.get_previous_active_version()
    assert previous is not None
    assert previous.version_id == "v2_a"

    rolled_back = registry.rollback_to_previous_version()
    assert rolled_back is not None
    assert rolled_back.version_id == "v2_a"
    event_types_after = [event.event_type for event in registry.list_registry_events()]
    assert "active_pointer_rollback" in event_types_after
    assert event_types_after.count("active_pointer_set") >= 3

    pointer_after = registry.get_active_pointer()
    assert pointer_after is not None
    assert pointer_after.version_id == "v2_a"
    assert pointer_after.previous_version_id == "v2_b"


def test_validate_rollback_target_requires_previous_active_version(tmp_path: Path, monkeypatch) -> None:
    registry = ModelRegistry(tmp_path / "registry")

    artifact_a = tmp_path / "artifact_a"
    artifact_a.mkdir(parents=True)
    artifact_b = tmp_path / "artifact_b"
    artifact_b.mkdir(parents=True)
    artifact_c = tmp_path / "artifact_c"
    artifact_c.mkdir(parents=True)

    registry.register_version("v2_a", artifact_a, metrics={"promotion_eligible": True})
    registry.register_version("v2_b", artifact_b, metrics={"promotion_eligible": True})
    registry.register_version("v2_c", artifact_c, metrics={"promotion_eligible": True})

    monkeypatch.setenv("QUANT_IMAGE", "registry.example/quant-bot@sha256:" + "a" * 64)
    save_model_bundle(train(*_sample_training_data(), horizon=4, calibration_frac=0.2), artifact_a / "model_4m.pkl")
    save_model_bundle(train(*_sample_training_data(), horizon=4, calibration_frac=0.2), artifact_b / "model_4m.pkl")
    save_model_bundle(train(*_sample_training_data(), horizon=4, calibration_frac=0.2), artifact_c / "model_4m.pkl")

    registry.set_active_version("v2_a")
    registry.set_active_version("v2_b")

    rolled_back = registry.validate_rollback_target("v2_a")
    assert rolled_back is not None
    assert rolled_back.version_id == "v2_a"

    with pytest.raises(ValueError, match="previous active version"):
        registry.validate_rollback_target("v2_c")


def test_registry_event_chain_tamper_is_detected(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    artifact = tmp_path / "artifact"
    artifact.mkdir(parents=True)

    registry.register_version("v2_a", artifact)
    registry.set_active_version("v2_a")

    events_path = registry.root / "registry_events.jsonl"
    content = events_path.read_text(encoding="utf-8")
    events_path.write_text(content.replace("v2_a", "v2_x", 1), encoding="utf-8")

    with pytest.raises(ValueError):
        registry.get_active_pointer()


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


def test_promote_version_updates_active_pointer_and_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    artifact_dir = tmp_path / "candidate"
    artifact_dir.mkdir(parents=True)
    _write_valid_model_bundle(artifact_dir, monkeypatch=monkeypatch)

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


def test_promote_version_records_artifact_contract_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    artifact_dir = tmp_path / "candidate"
    artifact_dir.mkdir(parents=True)
    _write_valid_model_bundle(artifact_dir, monkeypatch=monkeypatch)
    registry.register_version(
        "candidate_contract",
        artifact_dir,
        metrics={"promotion_eligible": True},
    )

    promoted = registry.promote_version(
        "candidate_contract",
        promoted_by="test",
        evidence_digest="sha256:" + "7" * 64,
        notes="approved",
    )
    assert promoted.status == "active"

    events = registry.list_registry_events()
    promotion_event = next(event for event in events if event.event_type == "promotion_recorded")
    assert promotion_event.payload["artifact_manifest_sha256"]
    assert promotion_event.payload["artifact_image_reference"].startswith("registry.example/quant-bot@sha256:")
    assert promotion_event.payload["artifact_feature_schema_sha256"]
    assert promotion_event.payload["artifact_threshold"] == "0.6"

    pointer = registry.get_active_pointer()
    assert pointer is not None
    assert pointer.artifact_manifest_sha256 == promotion_event.payload["artifact_manifest_sha256"]
    assert pointer.artifact_image_reference == promotion_event.payload["artifact_image_reference"]
    assert pointer.artifact_feature_schema_sha256 == promotion_event.payload["artifact_feature_schema_sha256"]
    assert pointer.artifact_threshold == promotion_event.payload["artifact_threshold"]


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


def test_promote_version_rejects_missing_manifest(tmp_path: Path, monkeypatch) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    artifact_dir = tmp_path / "candidate"
    artifact_dir.mkdir(parents=True)
    monkeypatch.setenv("QUANT_IMAGE", "registry.example/quant-bot@sha256:" + "a" * 64)
    (artifact_dir / "model_4m.pkl").write_text("placeholder", encoding="utf-8")
    registry.register_version(
        "candidate_missing_manifest",
        artifact_dir,
        metrics={"promotion_eligible": True},
    )

    with pytest.raises(ValueError):
        registry.promote_version("candidate_missing_manifest")


def test_promote_version_requires_two_distinct_approvals_when_enabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MODEL_REGISTRY_REQUIRE_TWO_PERSON_APPROVAL", "1")
    registry = ModelRegistry(tmp_path / "registry")
    artifact_dir = tmp_path / "candidate"
    artifact_dir.mkdir(parents=True)
    _write_valid_model_bundle(artifact_dir, monkeypatch=monkeypatch)
    registry.register_version(
        "candidate_two_person",
        artifact_dir,
        metrics={"promotion_eligible": True},
    )

    registry.record_promotion_approval(
        "candidate_two_person",
        approved_by="reviewer_a",
        evidence_digest="sha256:" + "1" * 64,
        reason="reviewed",
    )

    with pytest.raises(ValueError):
        registry.promote_version("candidate_two_person")

    registry.record_promotion_approval(
        "candidate_two_person",
        approved_by="reviewer_b",
        evidence_digest="sha256:" + "1" * 64,
        reason="reviewed",
    )

    promoted = registry.promote_version(
        "candidate_two_person",
        promoted_by="reviewer_a",
        evidence_digest="sha256:" + "1" * 64,
        notes="approved by two reviewers",
    )
    assert promoted.status == "active"
    assert registry.get_active_version() is not None


def test_promote_version_rejects_expired_or_mismatched_approvals_when_enabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MODEL_REGISTRY_REQUIRE_TWO_PERSON_APPROVAL", "1")
    registry = ModelRegistry(tmp_path / "registry")
    artifact_dir = tmp_path / "candidate"
    artifact_dir.mkdir(parents=True)
    _write_valid_model_bundle(artifact_dir, monkeypatch=monkeypatch)
    registry.register_version(
        "candidate_expired",
        artifact_dir,
        metrics={"promotion_eligible": True},
    )

    expired_at = datetime.now(timezone.utc).replace(year=datetime.now(timezone.utc).year - 1).isoformat()
    registry.record_promotion_approval(
        "candidate_expired",
        approved_by="reviewer_a",
        evidence_digest="sha256:" + "9" * 64,
        expires_at=expired_at,
        reason="reviewed",
    )
    registry.record_promotion_approval(
        "candidate_expired",
        approved_by="reviewer_b",
        evidence_digest="sha256:" + "9" * 64,
        reason="reviewed",
    )

    with pytest.raises(ValueError):
        registry.promote_version(
            "candidate_expired",
            evidence_digest="sha256:" + "9" * 64,
            promoted_by="reviewer_a",
        )


def test_terminal_registry_states_cannot_be_promoted(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    reject_artifact = tmp_path / "reject_artifact"
    reject_artifact.mkdir(parents=True)
    expire_artifact = tmp_path / "expire_artifact"
    expire_artifact.mkdir(parents=True)
    _write_valid_model_bundle(reject_artifact, monkeypatch=monkeypatch)
    _write_valid_model_bundle(expire_artifact, monkeypatch=monkeypatch)

    registry.register_version("candidate_rejected", reject_artifact, metrics={"promotion_eligible": True})
    registry.register_version("candidate_expired", expire_artifact, metrics={"promotion_eligible": True})
    registry.reject_version("candidate_rejected", notes="manual rejection")
    registry.expire_version("candidate_expired", notes="manual expiry")

    with pytest.raises(ValueError, match="not in promotable state"):
        registry.promote_version("candidate_rejected")
    with pytest.raises(ValueError, match="not in promotable state"):
        registry.promote_version("candidate_expired")


def test_mark_paper_quarantine_updates_candidate_status(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    artifact_dir = tmp_path / "candidate"
    artifact_dir.mkdir(parents=True)
    _write_valid_model_bundle(artifact_dir, monkeypatch=monkeypatch)
    registry.register_version("candidate_a", artifact_dir)

    record = registry.mark_paper_quarantine("candidate_a", notes="paper run")

    assert record.status == "paper_quarantine"
    assert record.promotion_notes == "paper run"


def test_paper_quarantine_requires_paper_evaluation_before_promotion(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    artifact_dir = tmp_path / "candidate"
    artifact_dir.mkdir(parents=True)
    _write_valid_model_bundle(artifact_dir, monkeypatch=monkeypatch)
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
