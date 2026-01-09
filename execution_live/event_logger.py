"""
Execution event logging utilities.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

from execution_live.order_models import ExecutionEvent


class ExecutionEventLogger:
    """
    In-memory logger with optional on-disk persistence and callback hooks.
    """

    def __init__(
        self,
        persist_path: Optional[str | Path] = None,
        sink: Optional[Callable[[ExecutionEvent], None]] = None,
    ):
        self._events: List[ExecutionEvent] = []
        self._sink = sink
        self._persist_path = Path(persist_path) if persist_path else None
        if self._persist_path:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event_type: str, payload: dict) -> ExecutionEvent:
        """
        Record an event and optionally forward to sink/persistence.
        """
        event = ExecutionEvent(
            event_type=event_type,
            timestamp=datetime.now(timezone.utc),
            payload=payload,
        )
        self._events.append(event)

        if self._sink:
            self._sink(event)

        if self._persist_path:
            with self._persist_path.open("a", encoding="utf-8") as handle:
                handle.write(event.model_dump_json() + "\n")

        return event

    def get_events(self) -> List[ExecutionEvent]:
        """Return a copy of recorded events."""
        return list(self._events)

    def clear(self) -> None:
        """Clear buffered events (does not truncate persisted file)."""
        self._events.clear()
