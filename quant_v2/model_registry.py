"""Simple model registry for v2 artifacts and active pointers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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


class ModelRegistry:
    """File-backed registry for v2 model versions."""

    def __init__(self, root: Path | str):
        self.root = Path(root).expanduser()
        self.versions_dir = self.root / "versions"
        self.active_file = self.root / "active.json"
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
        return updated

    def promote_version(
        self,
        version_id: str,
        *,
        promoted_by: str = "manual",
        notes: str = "",
    ) -> ModelVersionRecord:
        """Validate and activate a registered candidate model version."""

        clean_id = self._normalize_version_id(version_id)
        record = self.get_version(clean_id)
        if record is None:
            raise ValueError(f"Cannot promote unknown model version: {clean_id}")
        if not self._promotion_checks_pass(record):
            raise ValueError(f"Model version is not promotion eligible: {clean_id}")

        self.set_active_version(clean_id)
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
    ) -> None:
        """Point the registry to a previously registered active version."""

        clean_id = self._normalize_version_id(version_id)
        if self.get_version(clean_id) is None:
            raise ValueError(f"Cannot activate unknown model version: {clean_id}")

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
        }
        self._write_json_atomic(self.active_file, payload)
        self._mark_active_status(clean_id)

    def get_active_pointer(self) -> ActiveModelPointer | None:
        """Load active pointer metadata, including previous active version reference."""

        if not self.active_file.exists():
            return None

        payload = json.loads(self.active_file.read_text(encoding="utf-8"))
        version_id = payload.get("version_id")
        if not version_id:
            return None

        updated_at = str(payload.get("updated_at", ""))
        previous_version_id = payload.get("previous_version_id")
        if previous_version_id:
            previous_version_id = self._normalize_version_id(str(previous_version_id))
        else:
            previous_version_id = None

        return ActiveModelPointer(
            version_id=self._normalize_version_id(str(version_id)),
            updated_at=updated_at,
            previous_version_id=previous_version_id,
        )

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

        previous = self.get_version(pointer.previous_version_id)
        if previous is None:
            return None

        self.set_active_version(previous.version_id, previous_version_id=pointer.version_id)
        return previous

    def clear_active_version(self) -> None:
        """Remove active pointer from disk."""

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
        allowed = {"candidate", "paper_quarantine", "active", "inactive", "rejected"}
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
        if not (artifact_path / "config.json").exists() and not any(artifact_path.glob("model_*m.pkl")):
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

    def _version_file(self, version_id: str) -> Path:
        clean_id = self._normalize_version_id(version_id)
        return self.versions_dir / f"{clean_id}.json"

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(path)
