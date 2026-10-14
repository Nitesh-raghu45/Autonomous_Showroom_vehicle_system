"""Seniors Management Agent — empty stub.

Returns a placeholder message for all management / business-analysis queries.
This agent will be fleshed out with dashboards, pipeline reports, conversion
metrics, and forecasting in a future phase.
"""
from __future__ import annotations

import time
from typing import Any, AsyncIterator


STUB_MESSAGE = (
    "The Seniors Management dashboard is coming soon. "
    "This agent will provide business analysis, sales pipeline reports, "
    "conversion metrics, and forecasting capabilities. "
    "Please contact the management team directly for now."
)


async def handle(session_id: str, customer_number: str, question: str) -> AsyncIterator[dict[str, Any]]:
    """Yield a placeholder response — no real logic yet."""
    t0 = time.perf_counter()
    yield {
        "event": "meta",
        "data": {
            "customer_id": None,
            "customer_name": None,
            "route": "management",
            "intent": "MANAGEMENT_STUB",
            "reason": "management agent stub",
            "provider": None,
            "model": None,
            "history_turns": 0,
        },
    }
    yield {
        "event": "token",
        "data": {"text": STUB_MESSAGE},
    }
    total_ms = int((time.perf_counter() - t0) * 1000)
    yield {
        "event": "done",
        "data": {
            "source": "management_stub",
            "route": "management",
            "intent": "MANAGEMENT_STUB",
            "latency_ms": total_ms,
            "timings": {"total_ms": total_ms},
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "cached_prompt_tokens": 0},
            "provider": None,
            "model": None,
        },
    }
