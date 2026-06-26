"""Append-only audit sink helpers for observability artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AuditSinkRecord:
    """A single append-only audit record."""

    event_type: str
    created_at: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": str(self.event_type or ""),
            "created_at": str(self.created_at or ""),
            "payload": dict(self.payload),
        }


class JsonlAuditSink:
    """Minimal append-only JSONL sink for observability artifacts."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path).expanduser()

    def append(self, event_type: str, payload: dict[str, Any], *, created_at: str | None = None) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = AuditSinkRecord(
            event_type=event_type,
            created_at=created_at or datetime.now(timezone.utc).isoformat(),
            payload=dict(payload),
        )
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), sort_keys=True, default=str))
            handle.write("\n")
        return self.path
