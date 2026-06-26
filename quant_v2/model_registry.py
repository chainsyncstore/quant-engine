"""Simple model registry for v2 artifacts and active pointers."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_v2.models.trainer import load_model


@dataclass(frozen=True)
class ModelVersionRecord:
    """Model artifact metadata record."""

    version_id: str
    artifact_dir: str
    created_at: str
    metrics: dict[str, Any]
    tags: dict[str, Any]
    description: str = ""
    status: str = "candidate"
    promoted_at: str | None = None
    promoted_by: str | None = None
    promotion_notes: str = ""


@dataclass(frozen=True)
class ActiveModelPointer:
    """Current active version pointer plus previous active reference."""

    version_id: str
    updated_at: str
    previous_version_id: str | None = None
    artifact_manifest_sha256: str | None = None
    artifact_image_reference: str | None = None
    artifact_feature_schema_sha256: str | None = None
    artifact_dataset_digest: str | None = None
    artifact_threshold: str | None = None


@dataclass(frozen=True)
class RegistryEvent:
    """Append-only registry event used to derive the active pointer."""

    event_id: str
    event_type: str
    version_id: str | None
    created_at: str
    previous_event_hash: str | None
    event_hash: str
    payload: dict[str, Any] = field(default_factory=dict)


class ModelRegistry:
    """File-backed registry for v2 model versions."""

    def __init__(self, root: Path | str):
        self.root = Path(root).expanduser()
        self.versions_dir = self.root / "versions"
        self.active_file = self.root / "active.json"
        self.events_file = self.root / "registry_events.jsonl"
        self.versions_dir.mkdir(parents=True, exist_ok=True)

    def register_version(
        self,
        version_id: str,
        artifact_dir: Path | str,
        *,
        metrics: dict[str, Any] | None = None,
        tags: dict[str, Any] | None = None,
        description: str = "",
        status: str = "candidate",
    ) -> ModelVersionRecord:
        """Register a model version and persist its metadata."""

        clean_id = self._normalize_version_id(version_id)
        clean_status = self._normalize_status(status)
        artifact_path = Path(artifact_dir).expanduser().resolve()
        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact directory not found: {artifact_path}")

        record = ModelVersionRecord(
            version_id=clean_id,
            artifact_dir=str(artifact_path),
            created_at=datetime.now(timezone.utc).isoformat(),
            metrics=metrics or {},
            tags=tags or {},
            description=description,
            status=clean_status,
        )
        self._write_json_atomic(self._version_file(clean_id), asdict(record))
        self._append_event(
            "version_registered",
            version_id=clean_id,
            payload={
                "artifact_dir": str(artifact_path),
                "status": clean_status,
                "metrics": dict(record.metrics),
                "tags": dict(record.tags),
                "description": description,
            },
        )
        return record

    def get_version(self, version_id: str) -> ModelVersionRecord | None:
        """Load a registered version, if present."""

        path = self._version_file(version_id)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.setdefault("status", "candidate")
        payload.setdefault("promoted_at", None)
        payload.setdefault("promoted_by", None)
        payload.setdefault("promotion_notes", "")
        return ModelVersionRecord(**payload)

    def list_versions(self) -> list[ModelVersionRecord]:
        """Return all registered versions sorted by creation time."""

        records: list[ModelVersionRecord] = []
        for path in sorted(self.versions_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload.setdefault("status", "candidate")
            payload.setdefault("promoted_at", None)
            payload.setdefault("promoted_by", None)
            payload.setdefault("promotion_notes", "")
            records.append(ModelVersionRecord(**payload))
        return sorted(records, key=lambda r: r.created_at)

    def list_candidates(self) -> list[ModelVersionRecord]:
        """Return model versions that are not currently active."""

        return [
            record
            for record in self.list_versions()
            if record.status in {"candidate", "paper_quarantine"}
        ]

    def mark_paper_quarantine(
        self,
        version_id: str,
        *,
        notes: str = "",
    ) -> ModelVersionRecord:
        """Mark a registered candidate for paper-only quarantine evaluation."""

        return self.update_version_status(
            version_id,
            status="paper_quarantine",
            promotion_notes=notes,
        )

    def reject_version(
        self,
        version_id: str,
        *,
        notes: str = "",
    ) -> ModelVersionRecord:
        """Mark a registered version as rejected."""

        return self.update_version_status(
            version_id,
            status="rejected",
            promotion_notes=notes,
        )

    def expire_version(
        self,
        version_id: str,
        *,
        notes: str = "",
    ) -> ModelVersionRecord:
        """Mark a registered version as expired."""

        return self.update_version_status(
            version_id,
            status="expired",
            promotion_notes=notes,
        )

    def record_paper_evaluation(
        self,
        version_id: str,
        *,
        evaluation: dict[str, Any],
        promotion_eligible: bool,
        notes: str = "",
    ) -> ModelVersionRecord:
        """Attach forward paper-evaluation metrics to a quarantined model."""

        clean_id = self._normalize_version_id(version_id)
        record = self.get_version(clean_id)
        if record is None:
            raise ValueError(f"Unknown model version: {clean_id}")

        metrics = dict(record.metrics)
        metrics["paper_evaluation"] = {
            **dict(evaluation),
            "promotion_eligible": bool(promotion_eligible),
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }
        metrics["paper_quarantine_required"] = True
        updated = ModelVersionRecord(
            version_id=record.version_id,
            artifact_dir=record.artifact_dir,
            created_at=record.created_at,
            metrics=metrics,
            tags=dict(record.tags),
            description=record.description,
            status="candidate" if promotion_eligible else "paper_quarantine",
            promoted_at=record.promoted_at,
            promoted_by=record.promoted_by,
            promotion_notes=notes or record.promotion_notes,
        )
        self._write_json_atomic(self._version_file(clean_id), asdict(updated))
        return updated

    def validate_activation_ready(self, version_id: str) -> ModelVersionRecord:
        """Ensure a version can be safely activated with current runtime assets."""

        clean_id = self._normalize_version_id(version_id)
        record = self.get_version(clean_id)
        if record is None:
            raise ValueError(f"Unknown model version: {clean_id}")
        if record.status in {"rejected", "expired"}:
            raise ValueError(f"Model version is not reactivatable: {clean_id}")
        if not self._promotion_checks_pass(record):
            raise ValueError(f"Model version is not activation ready: {clean_id}")
        return record

    def validate_rollback_target(self, version_id: str) -> ModelVersionRecord:
        """Ensure a rollback target is the currently previous active version."""

        clean_id = self._normalize_version_id(version_id)
        record = self.validate_activation_ready(clean_id)
        pointer = self.get_active_pointer()
        if pointer is None or not pointer.previous_version_id:
            raise ValueError("No previous active version available for rollback")
        if pointer.previous_version_id != clean_id:
            raise ValueError(
                f"Rollback target must be the previous active version: {pointer.previous_version_id}"
            )
        return record

    def get_artifact_manifest(self, version_id: str) -> dict[str, Any]:
        """Load the validated artifact manifest for a registered version."""

        clean_id = self._normalize_version_id(version_id)
        record = self.get_version(clean_id)
        if record is None:
            raise ValueError(f"Unknown model version: {clean_id}")
        artifact_path = self._artifact_model_path(record)
        trained = load_model(artifact_path)
        manifest = dict(getattr(trained, "artifact_manifest", {}) or {})
        if not manifest:
            raise ValueError(f"Missing artifact manifest contents for: {clean_id}")
        return manifest

    def record_promotion_approval(
        self,
        version_id: str,
        *,
        approved_by: str,
        evidence_digest: str = "",
        scope: str = "live",
        reason: str = "",
        expires_at: str | None = None,
    ) -> RegistryEvent:
        """Record one append-only model promotion approval event."""

        clean_id = self._normalize_version_id(version_id)
        approver = str(approved_by).strip()
        if not approver:
            raise ValueError("approved_by cannot be empty")
        evidence = str(evidence_digest).strip()
        if not evidence:
            raise ValueError("evidence_digest cannot be empty")
        payload = {
            "approved_by": approver,
            "evidence_digest": evidence,
            "scope": str(scope).strip().lower() or "live",
            "reason": str(reason).strip(),
            "expires_at": expires_at,
        }
        return self._append_event(
            "promotion_approval_recorded",
            version_id=clean_id,
            payload=payload,
        )

    def update_version_status(
        self,
        version_id: str,
        *,
        status: str,
        promoted_at: str | None = None,
        promoted_by: str | None = None,
        promotion_notes: str | None = None,
    ) -> ModelVersionRecord:
        """Persist a status update for a registered version."""

        clean_id = self._normalize_version_id(version_id)
        clean_status = self._normalize_status(status)
        record = self.get_version(clean_id)
        if record is None:
            raise ValueError(f"Unknown model version: {clean_id}")

        updated = ModelVersionRecord(
            version_id=record.version_id,
            artifact_dir=record.artifact_dir,
            created_at=record.created_at,
            metrics=dict(record.metrics),
            tags=dict(record.tags),
            description=record.description,
            status=clean_status,
            promoted_at=promoted_at if promoted_at is not None else record.promoted_at,
            promoted_by=promoted_by if promoted_by is not None else record.promoted_by,
            promotion_notes=(
                promotion_notes
                if promotion_notes is not None
                else record.promotion_notes
            ),
        )
        self._write_json_atomic(self._version_file(clean_id), asdict(updated))
        self._append_event(
            "version_status_updated",
            version_id=clean_id,
            payload={
                "status": clean_status,
                "promoted_at": promoted_at,
                "promoted_by": promoted_by,
                "promotion_notes": promotion_notes,
            },
        )
        return updated

    def promote_version(
        self,
        version_id: str,
        *,
        promoted_by: str = "manual",
        evidence_digest: str = "",
        approval_scope: str = "live",
        notes: str = "",
    ) -> ModelVersionRecord:
        """Validate and activate a registered candidate model version."""

        clean_id = self._normalize_version_id(version_id)
        record = self.get_version(clean_id)
        if record is None:
            raise ValueError(f"Cannot promote unknown model version: {clean_id}")
        if record.status != "candidate":
            raise ValueError(f"Model version is not in promotable state: {clean_id}")
        if not self._promotion_checks_pass(record):
            raise ValueError(f"Model version is not promotion eligible: {clean_id}")
        clean_evidence_digest = str(evidence_digest).strip()
        clean_scope = str(approval_scope).strip().lower() or "live"
        contract_payload = self._artifact_contract_payload(record)
        if self._requires_two_person_approval():
            if not clean_evidence_digest:
                raise ValueError(
                    f"Model version requires an evidence digest for approval-gated promotion: {clean_id}"
                )
            self.validate_promotion_approvals(
                clean_id,
                evidence_digest=clean_evidence_digest,
                scope=clean_scope,
            )
        elif clean_evidence_digest:
            self._validate_approval_subset(
                clean_id,
                evidence_digest=clean_evidence_digest,
                scope=clean_scope,
            )

        self._append_event(
            "promotion_recorded",
            version_id=clean_id,
            payload={
                "promoted_by": promoted_by,
                "notes": notes,
                "evidence_digest": clean_evidence_digest,
                "approval_scope": clean_scope,
                **contract_payload,
            },
        )
        self.set_active_version(clean_id, contract_payload=contract_payload)
        return self.update_version_status(
            clean_id,
            status="active",
            promoted_at=datetime.now(timezone.utc).isoformat(),
            promoted_by=promoted_by,
            promotion_notes=notes,
        )

    def set_active_version(
        self,
        version_id: str,
        *,
        previous_version_id: str | None = None,
        contract_payload: dict[str, Any] | None = None,
    ) -> None:
        """Point the registry to a previously registered active version."""

        clean_id = self._normalize_version_id(version_id)
        record = self.get_version(clean_id)
        if record is None:
            raise ValueError(f"Cannot activate unknown model version: {clean_id}")
        if contract_payload is None:
            try:
                contract_payload = self._artifact_contract_payload(record)
            except Exception:
                contract_payload = {}
        else:
            contract_payload = dict(contract_payload)

        if previous_version_id is not None:
            if previous_version_id:
                previous_version_id = self._normalize_version_id(previous_version_id)
                if self.get_version(previous_version_id) is None:
                    raise ValueError(
                        f"Cannot set unknown previous model version: {previous_version_id}"
                    )
            else:
                previous_version_id = None
        else:
            current_pointer = self.get_active_pointer()
            if current_pointer is None:
                previous_version_id = None
            elif current_pointer.version_id != clean_id:
                previous_version_id = current_pointer.version_id
            else:
                previous_version_id = current_pointer.previous_version_id

        payload = {
            "version_id": clean_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "previous_version_id": previous_version_id,
            **contract_payload,
        }
        self._write_json_atomic(self.active_file, payload)
        self._append_event(
            "active_pointer_set",
            version_id=clean_id,
            payload={
                "previous_version_id": previous_version_id,
                **contract_payload,
            },
        )
        self._mark_active_status(clean_id)

    def get_active_pointer(self) -> ActiveModelPointer | None:
        """Load active pointer metadata, including previous active version reference."""

        return self._derive_active_pointer()

    def get_active_version(self) -> ModelVersionRecord | None:
        """Load the currently active version record."""

        pointer = self.get_active_pointer()
        if pointer is None:
            return None

        return self.get_version(pointer.version_id)

    def get_previous_active_version(self) -> ModelVersionRecord | None:
        """Return previously active version when available."""

        pointer = self.get_active_pointer()
        if pointer is None or not pointer.previous_version_id:
            return None

        return self.get_version(pointer.previous_version_id)

    def rollback_to_previous_version(self) -> ModelVersionRecord | None:
        """Switch active pointer to previous active version when available."""

        pointer = self.get_active_pointer()
        if pointer is None or not pointer.previous_version_id:
            return None

        previous = self.validate_rollback_target(pointer.previous_version_id)

        self._append_event(
            "active_pointer_rollback",
            version_id=previous.version_id,
            payload={"from_version_id": pointer.version_id},
        )
        self.set_active_version(previous.version_id, previous_version_id=pointer.version_id)
        return previous

    def clear_active_version(self) -> None:
        """Remove active pointer from disk."""

        self._append_event("active_pointer_cleared", version_id=None, payload={})
        if self.active_file.exists():
            self.active_file.unlink()

    @staticmethod
    def _normalize_version_id(version_id: str) -> str:
        clean = version_id.strip()
        if not clean:
            raise ValueError("version_id cannot be empty")
        if any(sep in clean for sep in ("/", "\\")):
            raise ValueError("version_id cannot contain path separators")
        return clean

    @staticmethod
    def _normalize_status(status: str) -> str:
        clean = str(status).strip().lower()
        allowed = {"candidate", "paper_quarantine", "active", "inactive", "rejected", "expired"}
        if clean not in allowed:
            raise ValueError(f"invalid model status: {status}")
        return clean

    @staticmethod
    def _promotion_checks_pass(record: ModelVersionRecord) -> bool:
        eligible = record.metrics.get("promotion_eligible")
        if eligible is False:
            return False
        if str(record.tags.get("promotion_eligible", "")).strip().lower() in {
            "0",
            "false",
            "no",
        }:
            return False
        artifact_path = Path(record.artifact_dir).expanduser()
        if not artifact_path.is_dir():
            return False
        model_paths = sorted(artifact_path.glob("model_*m.pkl"))
        if not model_paths:
            return False
        for model_path in model_paths:
            if not model_path.with_suffix(".manifest.json").exists():
                return False
            try:
                load_model(model_path)
            except Exception:
                return False
        if record.status == "paper_quarantine" or bool(record.metrics.get("paper_quarantine_required")):
            paper_eval = record.metrics.get("paper_evaluation")
            if not isinstance(paper_eval, dict):
                return False
            if paper_eval.get("promotion_eligible") is not True:
                return False
        return True

    def _mark_active_status(self, active_version_id: str) -> None:
        for record in self.list_versions():
            if record.version_id == active_version_id:
                continue
            if record.status == "active":
                self.update_version_status(record.version_id, status="inactive")
        active = self.get_version(active_version_id)
        if active is not None and active.status != "active":
            self.update_version_status(active_version_id, status="active")

    def _events(self) -> list[RegistryEvent]:
        if not self.events_file.exists():
            return []
        events: list[RegistryEvent] = []
        previous_hash: str | None = None
        for raw in self.events_file.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            event = RegistryEvent(
                event_id=str(payload.get("event_id", "")),
                event_type=str(payload.get("event_type", "")),
                version_id=payload.get("version_id"),
                created_at=str(payload.get("created_at", "")),
                previous_event_hash=payload.get("previous_event_hash"),
                event_hash=str(payload.get("event_hash", "")),
                payload=dict(payload.get("payload") or {}),
            )
            expected_previous = previous_hash
            if event.previous_event_hash != expected_previous:
                raise ValueError("registry event chain is broken")
            expected_event_id = hashlib.sha256(
                json.dumps(
                    {
                        "event_type": event.event_type,
                        "version_id": event.version_id,
                        "created_at": event.created_at,
                        "previous_event_hash": event.previous_event_hash,
                        "payload": event.payload,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
            ).hexdigest()
            expected_event_hash = hashlib.sha256(
                json.dumps(
                    {
                        "event_id": expected_event_id,
                        "event_type": event.event_type,
                        "version_id": event.version_id,
                        "created_at": event.created_at,
                        "previous_event_hash": event.previous_event_hash,
                        "payload": event.payload,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
            ).hexdigest()
            if event.event_id != expected_event_id or event.event_hash != expected_event_hash:
                raise ValueError("registry event hash mismatch")
            events.append(event)
            previous_hash = event.event_hash
        return events

    def list_registry_events(self) -> list[RegistryEvent]:
        """Return append-only registry events in order."""

        return self._events()

    def _append_event(
        self,
        event_type: str,
        *,
        version_id: str | None,
        payload: dict[str, Any],
    ) -> RegistryEvent:
        self.root.mkdir(parents=True, exist_ok=True)
        self.events_file.parent.mkdir(parents=True, exist_ok=True)
        events = self._events()
        previous_hash = events[-1].event_hash if events else None
        created_at = datetime.now(timezone.utc).isoformat()
        event_id = hashlib.sha256(
            json.dumps(
                {
                    "event_type": event_type,
                    "version_id": version_id,
                    "created_at": created_at,
                    "previous_event_hash": previous_hash,
                    "payload": payload,
                },
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        event_hash = hashlib.sha256(
            json.dumps(
                {
                    "event_id": event_id,
                    "event_type": event_type,
                    "version_id": version_id,
                    "created_at": created_at,
                    "previous_event_hash": previous_hash,
                    "payload": payload,
                },
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        event = RegistryEvent(
            event_id=event_id,
            event_type=event_type,
            version_id=version_id,
            created_at=created_at,
            previous_event_hash=previous_hash,
            event_hash=event_hash,
            payload=dict(payload),
        )
        with self.events_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(event), sort_keys=True, default=str))
            handle.write("\n")
        return event

    def _derive_active_pointer(self) -> ActiveModelPointer | None:
        pointer: ActiveModelPointer | None = None
        for event in self._events():
            if event.event_type in {"active_pointer_set", "active_pointer_rollback"}:
                previous_version_id = event.payload.get("previous_version_id")
                if previous_version_id:
                    previous_version_id = self._normalize_version_id(str(previous_version_id))
                else:
                    previous_version_id = pointer.version_id if pointer is not None else None
                if event.version_id:
                    pointer = ActiveModelPointer(
                        version_id=self._normalize_version_id(str(event.version_id)),
                        updated_at=event.created_at,
                        previous_version_id=previous_version_id,
                        artifact_manifest_sha256=self._normalize_optional_contract_value(
                            event.payload.get("artifact_manifest_sha256")
                        ),
                        artifact_image_reference=self._normalize_optional_contract_value(
                            event.payload.get("artifact_image_reference")
                        ),
                        artifact_feature_schema_sha256=self._normalize_optional_contract_value(
                            event.payload.get("artifact_feature_schema_sha256")
                        ),
                        artifact_dataset_digest=self._normalize_optional_contract_value(
                            event.payload.get("artifact_dataset_digest")
                        ),
                        artifact_threshold=self._normalize_optional_contract_value(
                            event.payload.get("artifact_threshold")
                        ),
                    )
            elif event.event_type == "active_pointer_cleared":
                pointer = None
        return pointer

    def _promotion_approval_events(self, version_id: str) -> list[RegistryEvent]:
        clean_id = self._normalize_version_id(version_id)
        return [
            event
            for event in self._events()
            if event.event_type == "promotion_approval_recorded" and self._normalize_version_id(str(event.version_id or "")) == clean_id
        ]

    def _promotion_approval_events_for(
        self,
        version_id: str,
        *,
        evidence_digest: str = "",
        scope: str = "live",
    ) -> list[RegistryEvent]:
        clean_id = self._normalize_version_id(version_id)
        clean_scope = str(scope).strip().lower() or "live"
        clean_evidence = str(evidence_digest).strip()
        events = self._promotion_approval_events(clean_id)
        filtered: list[RegistryEvent] = []
        for event in events:
            if not self._approval_event_is_valid(
                event,
                scope=clean_scope,
                evidence_digest=clean_evidence,
            ):
                continue
            filtered.append(event)
        return filtered

    def _approval_event_is_valid(
        self,
        event: RegistryEvent,
        *,
        evidence_digest: str,
        scope: str,
    ) -> bool:
        payload = event.payload or {}
        if str(payload.get("approved_by") or "").strip() == "":
            return False
        if str(payload.get("scope") or "live").strip().lower() != scope:
            return False
        if str(payload.get("evidence_digest") or "").strip() != evidence_digest:
            return False
        expires_at = str(payload.get("expires_at") or "").strip()
        if not expires_at:
            return True
        try:
            parsed = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
        except ValueError:
            return False
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed >= datetime.now(timezone.utc)

    @staticmethod
    def _requires_two_person_approval() -> bool:
        raw = str(os.getenv("MODEL_REGISTRY_REQUIRE_TWO_PERSON_APPROVAL", "0")).strip().lower()
        return raw in {"1", "true", "yes", "on"}

    def _has_two_distinct_approvals(
        self,
        version_id: str,
        *,
        evidence_digest: str = "",
        scope: str = "live",
    ) -> bool:
        approvals = self._promotion_approval_events_for(
            version_id,
            evidence_digest=evidence_digest,
            scope=scope,
        )
        approved_by: list[str] = []
        for event in approvals:
            approver = str(event.payload.get("approved_by") or "").strip()
            if approver and approver not in approved_by:
                approved_by.append(approver)
        return len(approved_by) >= 2

    def validate_promotion_approvals(
        self,
        version_id: str,
        *,
        evidence_digest: str,
        scope: str = "live",
    ) -> list[RegistryEvent]:
        """Validate approval evidence for a live promotion."""

        clean_id = self._normalize_version_id(version_id)
        clean_evidence = str(evidence_digest).strip()
        if not clean_evidence:
            raise ValueError(f"evidence_digest cannot be empty: {clean_id}")
        clean_scope = str(scope).strip().lower() or "live"
        approvals = self._promotion_approval_events_for(
            clean_id,
            evidence_digest=clean_evidence,
            scope=clean_scope,
        )
        if not approvals:
            raise ValueError(
                f"Model version is missing matching promotion approvals for evidence digest: {clean_id}"
            )
        if not self._has_two_distinct_approvals(
            clean_id,
            evidence_digest=clean_evidence,
            scope=clean_scope,
        ):
            raise ValueError(
                f"Model version is missing two distinct promotion approvals for evidence digest: {clean_id}"
            )
        return approvals

    def _validate_approval_subset(
        self,
        version_id: str,
        *,
        evidence_digest: str,
        scope: str = "live",
    ) -> list[RegistryEvent]:
        """Validate that any provided approvals match the promoted evidence."""

        return self._promotion_approval_events_for(
            version_id,
            evidence_digest=evidence_digest,
            scope=scope,
        )

    def _version_file(self, version_id: str) -> Path:
        clean_id = self._normalize_version_id(version_id)
        return self.versions_dir / f"{clean_id}.json"

    def _artifact_model_path(self, record: ModelVersionRecord) -> Path:
        artifact_path = Path(record.artifact_dir).expanduser()
        model_paths = sorted(artifact_path.glob("model_*m.pkl"))
        if not model_paths:
            raise ValueError(f"Model version has no saved model artifacts: {record.version_id}")
        return model_paths[0]

    def _artifact_contract_payload(self, record: ModelVersionRecord) -> dict[str, str]:
        manifest = self.get_artifact_manifest(record.version_id)
        threshold_policy = (manifest.get("training") or {}).get("threshold_policy") or {}
        selected_threshold = threshold_policy.get("selected_threshold")
        if selected_threshold is None:
            selected_threshold = (manifest.get("training") or {}).get("threshold")
        return {
            "artifact_manifest_sha256": str(
                (manifest.get("checksums") or {}).get("artifact_manifest_sha256") or ""
            ),
            "artifact_image_reference": str(
                (manifest.get("runtime") or {}).get("image_reference") or ""
            ),
            "artifact_feature_schema_sha256": str(
                (manifest.get("model") or {}).get("feature_schema_sha256") or ""
            ),
            "artifact_dataset_digest": str(
                (manifest.get("training") or {}).get("dataset_digest") or ""
            ),
            "artifact_threshold": str(selected_threshold or ""),
        }

    @staticmethod
    def _normalize_optional_contract_value(value: object) -> str | None:
        clean = str(value).strip()
        return clean or None

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(path)
