from __future__ import annotations

import json
import logging
from typing import Dict, List

from .types import SourceCallLog


class RetrievalStructuredLogger:
    """Structured logger for source-level retrieval diagnostics."""

    def __init__(self, logger_name: str = "graph_rag.retrieval"):
        self._logger = logging.getLogger(logger_name)

    def log_call(self, call: SourceCallLog) -> None:
        self._logger.info("RETRIEVAL_CALL %s", json.dumps(call.to_dict(), ensure_ascii=True))

    def log_summary(self, query: str, counts: Dict[str, int], calls: List[SourceCallLog]) -> None:
        payload = {
            "query": query,
            "source_counts": counts,
            "calls": [c.to_dict() for c in calls],
        }
        self._logger.info("RETRIEVAL_SUMMARY %s", json.dumps(payload, ensure_ascii=True))
