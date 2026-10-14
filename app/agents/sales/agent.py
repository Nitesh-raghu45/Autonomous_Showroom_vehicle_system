"""Sales Agent — delegates entirely to the existing LangGraph workflow.

This is a zero-logic pass-through: every event from ``app.graph.run_chat``
is yielded unchanged. The existing Sales Agent code (graph, router, actions,
planner, prompting) is never modified.
"""
from __future__ import annotations

from typing import Any, AsyncIterator

from ...graph import run_chat


async def handle(session_id: str, customer_number: str, question: str) -> AsyncIterator[dict[str, Any]]:
    """Run the existing Sales Agent graph and yield SSE events unchanged."""
    async for event in run_chat(session_id, customer_number, question):
        yield event
