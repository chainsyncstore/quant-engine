"""
Utilities that connect ExecutionIntent streams to adapters.
"""

from __future__ import annotations

from typing import List, Optional

from execution_live.adapter import ExecutionAdapter
from execution_live.order_models import ExecutionIntent, ExecutionReport


class PaperExecutionService:
    """
    Thin orchestration layer that forwards intents to an adapter and stores reports.
    """

    def __init__(self, adapter: ExecutionAdapter):
        self._adapter = adapter
        self._reports: List[ExecutionReport] = []

    def handle_intent(self, intent: ExecutionIntent) -> ExecutionReport:
        """
        Forward a single intent to the adapter and store the resulting report.
        """
        report = self._adapter.place_order(intent)
        self._reports.append(report)
        return report

    @property
    def reports(self) -> List[ExecutionReport]:
        return list(self._reports)

    def latest_report(self) -> Optional[ExecutionReport]:
        return self._reports[-1] if self._reports else None
