"""Supervisor Router — classifies user messages to domain-specific agents.

Strategy: deterministic keyword matching first (microseconds, zero LLM overhead).
If no service/management keywords match, the message goes to the Sales Agent
(the existing default), preserving 100% backward compatibility.

Session affinity: once an agent is selected for a session turn, the session
remembers ``active_agent`` so follow-up messages (slot fills, confirmations)
stay with the correct agent — unless the user clearly switches context.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any, AsyncIterator

from .. import cache
from .registry import AgentDomain
from .sales.agent import handle as sales_handle
from .service.agent import handle as service_handle
from .management.agent import handle as management_handle

log = logging.getLogger("supervisor")


# ---------------------------------------------------------------------------
# Keyword patterns (longest-first matching, case-insensitive)
# ---------------------------------------------------------------------------

_SERVICE_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        # Explicit service / maintenance phrases
        r"\bservice\s+appointment",
        r"\bservice\s+booking",
        # verb and "service" must be close together - a greedy `.*` let
        # "book a test drive, great service btw" reach the service agent
        r"\b(?:book|schedule|reschedule|cancel|postpone)\b(?:\s+[\w']+){0,4}?\s+service\b",
        r"\bcar\s+service\b",
        r"\bvehicle\s+service\b",
        r"\bservice\s+center\b",
        r"\bservice\s+schedule\b",
        r"\bservice\s+slot",
        r"\bservice\s+history\b",
        r"\bservice\s+status\b",
        r"\bnext\s+service\b",
        r"\bfree\s+service\b",
        r"\bpaid\s+service\b",
        r"\bperiodic\s+maintenance\b",
        r"\broutine\s+maintenance\b",
        r"\bscheduled\s+maintenance\b",
        r"\bservice\s+due\b",
        # Specific service actions
        r"\boil\s+change\b",
        r"\btyre\s+rotation\b",
        r"\btire\s+rotation\b",
        r"\bwheel\s+alignment\b",
        r"\bbrake\s+(check|service|pad)",
        r"\bbattery\s+(check|service|replace)",
        r"\bac\s+(service|repair|check)\b",
        # "Does the Creta have air conditioning?" is a sales feature question - only
        # an AC *problem / repair* is a service request.
        r"\b(?:ac|a/c|air[\s-]?con\w*)\b(?:\s+[\w']+){0,4}?\s+(?:not working|stopped|repair\w*|servic\w*|check\w*|problem|issue|gas|leak\w*|fix\w*|not cooling|isn'?t cooling)\b",
        r"\b(?:repair|fix|check|service)\b(?:\s+[\w']+){0,3}?\s+(?:ac|a/c|air[\s-]?con\w*)\b",
        r"\bbody\s+repair\b",
        r"\bdent(ing)?\s+(and\s+)?paint",
        r"\baccident(al)?\s+repair\b",
        r"\bcollision\s+repair\b",
        r"\bwarranty\s+(claim|service|repair)\b",
        # Service appointment IDs
        r"\bSVC-[A-Z0-9]+",
        r"\bservice\s+appointment\s+id\b",
    ]
]

_MANAGEMENT_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bsales\s+report\b",
        r"\brevenue\s+(report|analysis|trend)",
        r"\bpipeline\s+(report|analysis|status)",
        r"\bconversion\s+rate\b",
        r"\bmonthly\s+target\b",
        r"\bbusiness\s+analysis\b",
        r"\bmanagement\s+(report|dashboard)\b",
        r"\bforecast\b",
        r"\bperformance\s+(report|metric|kpi)\b",
        r"\bkpi\b",
        r"\banalytics\s+dashboard\b",
        r"\bsenior\s+management\b",
        r"\bdealership\s+performance\b",
        r"\bsales\s+funnel\b",
        r"\blead\s+conversion\b",
    ]
]


def _match_domain(question: str) -> AgentDomain | None:
    """Check keyword patterns. Returns None if no pattern matches."""
    for pat in _SERVICE_PATTERNS:
        if pat.search(question):
            return AgentDomain.SERVICE
    for pat in _MANAGEMENT_PATTERNS:
        if pat.search(question):
            return AgentDomain.MANAGEMENT
    return None


# ---------------------------------------------------------------------------
# Session affinity
# ---------------------------------------------------------------------------

async def _get_active_agent(session_id: str) -> str | None:
    """Read the active_agent field from the main session state."""
    if not session_id:
        return None
    r = cache._client
    if r is None:
        return None
    try:
        import json
        raw = await r.get(f"sst:{session_id}")
        if raw:
            state = json.loads(raw)
            return state.get("active_agent")
    except Exception:
        pass
    return None


async def _set_active_agent(session_id: str, agent: str) -> None:
    """Persist the active_agent field into the main session state."""
    if not session_id:
        return
    r = cache._client
    if r is None:
        return
    try:
        import json
        raw = await r.get(f"sst:{session_id}")
        state = json.loads(raw) if raw else {}
        state["active_agent"] = agent
        await r.set(f"sst:{session_id}", json.dumps(state), ex=3600)
    except Exception:
        log.debug("couldn't persist active_agent for session %s", session_id)


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

async def classify_and_dispatch(
    session_id: str, customer_number: str, question: str
) -> AsyncIterator[dict[str, Any]]:
    """Classify the user's message and route to the appropriate agent.

    Yields SSE events (the same shape all three agents produce).
    """
    t0 = time.perf_counter()

    # 1. Try keyword classification on the current message
    domain = _match_domain(question)

    # 2. If no keywords matched, check session affinity (stay with current agent
    #    if we're in a multi-turn flow like slot filling)
    if domain is None:
        prev_agent = await _get_active_agent(session_id)
        if prev_agent == AgentDomain.SERVICE.value:
            # Check if there's a pending service action (multi-turn slot fill)
            from .service.actions import load_service_session
            svc_session = await load_service_session(session_id)
            if svc_session and svc_session.get("pending"):
                domain = AgentDomain.SERVICE
                log.info("supervisor: staying with service agent (pending slot fill)")

    # 3. Default → Sales Agent (backward-compatible)
    if domain is None:
        domain = AgentDomain.SALES

    classify_ms = int((time.perf_counter() - t0) * 1000)
    log.info(
        "supervisor: session_id=%s domain=%s classify_ms=%d question=%r",
        session_id, domain.value, classify_ms, question[:80],
    )

    # 4. Persist agent affinity
    await _set_active_agent(session_id, domain.value)

    # 5. Dispatch
    if domain == AgentDomain.SERVICE:
        async for event in service_handle(session_id, customer_number, question):
            yield event
    elif domain == AgentDomain.MANAGEMENT:
        async for event in management_handle(session_id, customer_number, question):
            yield event
    else:
        async for event in sales_handle(session_id, customer_number, question):
            yield event
