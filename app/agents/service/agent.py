"""Service Agent — LangGraph sub-graph for vehicle service appointment management.

Sub-graph topology:

    START
      │
    load_service_state ──(error)──► END
      │
    service_route          (deterministic intent classification)
      │
    service_handler        (slot-fill + execute action)
      │
    save_service_history
      │
    END

The agent yields SSE-style events (meta → token* → [action] → done) identical
to the Sales Agent, so the HTTP layer and React client are unchanged.
"""
from __future__ import annotations

import logging
import re
import time
from datetime import date
from typing import Any, AsyncIterator

from ... import cache, clock
from ...config import settings
from ..registry import AgentDomain
from . import intents as SI
from .actions import (
    ServiceResult,
    book_service,
    cancel_service,
    check_available_slots,
    list_service_appts,
    load_service_session,
    reschedule_service,
    save_service_session,
    service_history,
)
from .slots import extract_all_slots, extract_service_appt_id

log = logging.getLogger("service.agent")

# Sentinel values matching the sales agent's guest detection
_GUEST = {"", "guest", "none", "null"}


# ---------------------------------------------------------------------------
# Intent routing (deterministic, no LLM)
# ---------------------------------------------------------------------------

_BOOK_RE = re.compile(
    r"\b(book|schedule|arrange|set up|fix|want)\b.*\b(service|maintenance|repair|oil change|tyre|tire|brake|battery|ac |air con|wheel align)",
    re.IGNORECASE,
)
_RESCHEDULE_RE = re.compile(
    r"\b(reschedule|postpone|prepone|move|shift|change.*(?:date|time|slot))\b.*\b(service|maintenance|repair|appointment|SVC-)",
    re.IGNORECASE,
)
_CANCEL_RE = re.compile(
    r"\b(cancel|drop|remove|delete)\b.*\b(service|maintenance|repair|appointment|SVC-)",
    re.IGNORECASE,
)
_SLOTS_RE = re.compile(
    r"\b(available|open|free)\b.*\b(slot|time|appointment|service)\b|\b(slot|time)s?\b.*\b(available|open|free)\b|\bcheck.*slot|\bshow.*slot",
    re.IGNORECASE,
)
_HISTORY_RE = re.compile(
    r"\b(service\s+history|past\s+service|previous\s+service|service\s+record|my\s+service)",
    re.IGNORECASE,
)
_STATUS_RE = re.compile(
    r"\b(status|track|where)\b.*\b(service|SVC-)",
    re.IGNORECASE,
)
_YES_RE = re.compile(r"^\s*(yes|yep|yeah|sure|ok|okay|confirm|go ahead|do it|proceed|haan|ha)\s*[.!]?\s*$", re.IGNORECASE)
_NO_RE = re.compile(r"^\s*(no|nope|nah|cancel|don'?t|stop|never mind|nahi|mat)\s*[.!]?\s*$", re.IGNORECASE)


def _classify_intent(question: str, pending: dict[str, Any] | None) -> SI.ServiceRoute:
    """Classify a user message into a service intent."""
    q = question.strip()

    # Handle confirmation flow
    if pending:
        if _YES_RE.match(q):
            return SI.ServiceRoute(
                intent=pending.get("type", SI.BOOK_SERVICE),
                slots={"confirmed": True, **pending},
                reason="user confirmed pending action",
            )
        if _NO_RE.match(q):
            return SI.ServiceRoute(
                intent=pending.get("type", SI.BOOK_SERVICE),
                slots={"declined": True},
                reason="user declined pending action",
            )
        # If there's a pending booking and user provides more slots (date/time/type),
        # treat it as continued slot-filling
        if pending.get("type") in ("BOOK_SERVICE", "RESCHEDULE_SERVICE"):
            return SI.ServiceRoute(
                intent=pending["type"],
                slots=pending,
                reason="continuing slot fill from pending",
            )

    if _CANCEL_RE.search(q):
        return SI.ServiceRoute(SI.CANCEL_SERVICE, reason="cancel pattern match")
    if _RESCHEDULE_RE.search(q):
        return SI.ServiceRoute(SI.RESCHEDULE_SERVICE, reason="reschedule pattern match")
    if _SLOTS_RE.search(q):
        return SI.ServiceRoute(SI.CHECK_SLOTS, reason="slot availability query")
    if _HISTORY_RE.search(q):
        return SI.ServiceRoute(SI.SERVICE_HISTORY, reason="service history query")
    if _STATUS_RE.search(q):
        return SI.ServiceRoute(SI.SERVICE_STATUS, reason="service status query")
    if _BOOK_RE.search(q):
        return SI.ServiceRoute(SI.BOOK_SERVICE, reason="booking pattern match")

    # Default: treat as booking intent (the supervisor already classified this as service-domain)
    return SI.ServiceRoute(SI.BOOK_SERVICE, reason="default service intent")


# ---------------------------------------------------------------------------
# Action execution
# ---------------------------------------------------------------------------

async def _execute(
    intent: str,
    slots: dict[str, Any],
    extracted: dict[str, Any],
    cid: str | None,
    pending: dict[str, Any] | None,
) -> ServiceResult:
    """Execute the classified service intent."""

    # Guest guard
    if not cid and intent not in (SI.CHECK_SLOTS, SI.SERVICE_GREETING):
        return ServiceResult(
            "To manage service appointments, I need to verify your identity first. "
            "Please start a session with your customer number or mobile number."
        )

    if intent == SI.BOOK_SERVICE:
        if slots.get("declined"):
            return ServiceResult("Okay, I've left everything as it is.", clear_pending=True)
        if slots.get("confirmed") and pending and pending.get("type") == "CANCEL_SERVICE":
            return await cancel_service(cid, pending.get("service_appt_id"), confirmed=True)
        return await book_service(
            cid=cid,
            service_type=extracted.get("service_type"),
            date_str=extracted.get("date"),
            hour=extracted.get("hour"),
            minute=extracted.get("minute"),
            vehicle_reg_no=extracted.get("vehicle_reg_no"),
            notes=extracted.get("notes"),
            center=extracted.get("center", "default"),
            date_error=extracted.get("date_error"),
            time_error=extracted.get("time_error"),
        )

    if intent == SI.RESCHEDULE_SERVICE:
        if slots.get("declined"):
            return ServiceResult("Okay, I've left everything as it is.", clear_pending=True)
        return await reschedule_service(
            cid=cid,
            appt_id=extracted.get("service_appt_id"),
            new_date_str=extracted.get("date"),
            new_hour=extracted.get("hour"),
            new_minute=extracted.get("minute"),
        )

    if intent == SI.CANCEL_SERVICE:
        if slots.get("declined"):
            return ServiceResult("Okay, I've left everything as it is.", clear_pending=True)
        confirmed = slots.get("confirmed", False)
        appt_id = extracted.get("service_appt_id")
        if confirmed and pending:
            appt_id = pending.get("service_appt_id", appt_id)
        return await cancel_service(cid=cid, appt_id=appt_id, confirmed=confirmed)

    if intent == SI.CHECK_SLOTS:
        dt = None
        if extracted.get("date"):
            try:
                dt = date.fromisoformat(extracted["date"])
            except ValueError:
                pass
        if dt is None:
            dt = clock.today()
        center = extracted.get("center", "default")
        return await check_available_slots(center, dt)

    if intent == SI.SERVICE_HISTORY:
        return await service_history(cid)

    if intent == SI.SERVICE_STATUS:
        appt_id = extracted.get("service_appt_id")
        if not appt_id:
            return await service_history(cid)
        from .actions import _get_service_appt
        appt = await _get_service_appt(cid, appt_id)
        if appt is None:
            return ServiceResult(f"I couldn't find service appointment {appt_id} on your account.")
        from .models import SERVICE_TYPE_LABEL
        label = SERVICE_TYPE_LABEL.get(appt.get("service_type", ""), "Service")
        return ServiceResult(
            f"{label} on {appt['date']} at {appt['time']} — Status: {appt.get('status', 'unknown')} "
            f"(ID: {appt['service_appt_id']})"
        )

    return ServiceResult(
        "I'm not sure what you'd like to do. I can help you book, reschedule, "
        "or cancel a service appointment, or check available slots."
    )


# ---------------------------------------------------------------------------
# Main handler — yields SSE events
# ---------------------------------------------------------------------------

async def handle(session_id: str, customer_number: str, question: str) -> AsyncIterator[dict[str, Any]]:
    """Service Agent entry point — yields SSE events compatible with the chat stream."""
    t0 = time.perf_counter()

    # --- Resolve customer ---
    number = (customer_number or "").strip()
    cid: str | None = None

    if session_id:
        cid = await cache.session_customer(session_id)
    if not cid and number.lower() not in _GUEST:
        cid = await cache.resolve_number(number)

    customer_name = None
    if cid:
        record = await cache.get_customer_record(cid)
        if record:
            customer_name = (record.get("customer") or {}).get("name")

    # --- Load service session state ---
    svc_session = await load_service_session(session_id) or {}
    pending = svc_session.get("pending")

    # --- Route ---
    route = _classify_intent(question, pending)
    route_ms = int((time.perf_counter() - t0) * 1000)

    log.info(
        "service.agent: session_id=%s cid=%s intent=%s reason=%s route_ms=%d",
        session_id, cid, route.intent, route.reason, route_ms,
    )

    yield {
        "event": "meta",
        "data": {
            "customer_id": cid,
            "customer_name": customer_name,
            "route": "service",
            "intent": route.intent,
            "reason": route.reason,
            "provider": None,
            "model": None,
            "history_turns": len(svc_session.get("turns", [])) // 2,
        },
    }

    # --- Extract slots and execute ---
    extracted = extract_all_slots(question, pending)
    try:
        result = await _execute(route.intent, route.slots, extracted, cid, pending)
    except Exception:
        log.exception("service agent handler failed: session_id=%s intent=%s", session_id, route.intent)
        result = ServiceResult("Sorry, something went wrong with the service agent. Please try again.")

    # --- Emit token ---
    yield {"event": "token", "data": {"text": result.text}}

    # --- Emit action if any ---
    if result.action:
        yield {"event": "action", "data": result.action}

    # --- Update service session state ---
    if result.clear_pending:
        svc_session["pending"] = None
    elif result.pending:
        svc_session["pending"] = result.pending
    elif pending:
        pending["turns_left"] = int(pending.get("turns_left", 1)) - 1
        svc_session["pending"] = pending if pending["turns_left"] > 0 else None

    # Append turn to service-specific history
    turns = svc_session.get("turns", [])
    turns.append(["user", question])
    turns.append(["assistant", result.text])
    svc_session["turns"] = turns[-24:]   # keep last 12 turn pairs
    await save_service_session(session_id, svc_session)

    # --- Done ---
    total_ms = int((time.perf_counter() - t0) * 1000)
    yield {
        "event": "done",
        "data": {
            "source": "service_agent",
            "route": "service",
            "intent": route.intent,
            "latency_ms": total_ms,
            "timings": {"route_ms": route_ms, "total_ms": total_ms},
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "cached_prompt_tokens": 0},
            "provider": None,
            "model": None,
        },
    }
