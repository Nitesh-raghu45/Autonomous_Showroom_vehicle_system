"""Chat orchestration - public entry point of the request hot path.

The Supervisor Router (``app.agents.supervisor``) classifies the user's message
and dispatches to the correct domain agent:

    Sales Agent (default)  →  existing ``app/graph/run_chat`` (zero changes)
    Service Agent          →  ``app/agents/service`` (new: book/reschedule/cancel service)
    Management Agent       →  ``app/agents/management`` (stub — coming soon)

``answer()`` keeps its signature and event names (meta → token* → [replace] → [action] → done | error),
so the HTTP layer and the React client are unchanged.
"""
from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from .graph import run_chat                                                  # noqa: F401 (used by sales agent wrapper)
from .graph.nodes import FALLBACK_ERROR, FALLBACK_TIMEOUT, GUEST, REPLAY_CHUNK  # noqa: F401  (re-exported)
from .llm import ProviderError, get_provider
from .agents.supervisor import classify_and_dispatch

log = logging.getLogger("service")


class UnknownCustomer(LookupError):
    pass


async def answer(session_id: str, customer_number: str, question: str) -> AsyncIterator[dict[str, Any]]:
    async for event in classify_and_dispatch(session_id, customer_number, question):
        yield event


async def warm_llm() -> None:
    """Open the LLM connection and prime the server's prefix cache with the static rules."""
    from .prompting import SYSTEM_RULES
    try:
        await get_provider().warmup([{"role": "system", "content": SYSTEM_RULES}, {"role": "user", "content": "ping"}])
    except ProviderError:
        pass