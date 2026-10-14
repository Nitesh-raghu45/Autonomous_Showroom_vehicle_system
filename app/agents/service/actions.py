"""Service Agent business actions — book, reschedule, cancel service appointments.

Mirrors the pattern of ``app.actions`` but operates exclusively on the
``svc:`` Redis namespace. The Sales Agent's ``app.actions`` is never touched.
"""
from __future__ import annotations

import json
import logging
import time as _time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

from pydantic import ValidationError

from ... import cache, clock, persistence
from .models import (
    SERVICE_TYPE_LABEL,
    ServiceAppointmentRequest,
    ServiceAppointmentStatus,
    ServiceRescheduleRequest,
    ServiceType,
)
from . import slot_store

log = logging.getLogger("service.actions")

PENDING_TTL_TURNS = 3

# Redis key helpers
_APPT_KEY = "svc:appt:{cid}:{appt_id}"
_APPTS_SET_KEY = "svc:appts:{cid}"
_SESSION_KEY = "svc:session:{session_id}"


# ---------------------------------------------------------------------------
# Internal Redis access — same pattern as slot_store
# ---------------------------------------------------------------------------

def _get_client():
    return cache._client


async def _safe(coro, default=None):
    client = _get_client()
    if client is None or _time.monotonic() < cache._down_until:
        coro.close()
        return default
    try:
        return await coro
    except Exception as exc:
        log.warning("service.actions redis op failed: %s", exc)
        cache._trip()
        return default


# ---------------------------------------------------------------------------
# Result model (mirrors app.actions.Result)
# ---------------------------------------------------------------------------

@dataclass
class ServiceResult:
    text: str
    action: dict[str, Any] | None = None
    mutated: bool = False
    pending: dict[str, Any] | None = None
    clear_pending: bool = False


# ---------------------------------------------------------------------------
# Redis helpers
# ---------------------------------------------------------------------------

async def _save_service_appt(cid: str, appt: dict[str, Any]) -> None:
    """Persist a service appointment to Redis."""
    r = _get_client()
    if r is None:
        raise cache.StoreUnavailable("Redis unavailable for service appointment save")
    appt_id = appt["service_appt_id"]
    key = _APPT_KEY.format(cid=cid, appt_id=appt_id)
    await _safe(r.set(key, json.dumps(appt), ex=86400 * 90))  # keep 90 days
    await _safe(r.sadd(_APPTS_SET_KEY.format(cid=cid), appt_id))


async def _get_service_appt(cid: str, appt_id: str) -> dict[str, Any] | None:
    """Load a single service appointment."""
    r = _get_client()
    if r is None:
        return None
    key = _APPT_KEY.format(cid=cid, appt_id=appt_id)
    raw = await _safe(r.get(key))
    if raw is None:
        return None
    return json.loads(raw)


async def list_service_appts(cid: str) -> list[dict[str, Any]]:
    """Load all service appointments for a customer."""
    r = _get_client()
    if r is None:
        return []
    ids_raw = await _safe(r.smembers(_APPTS_SET_KEY.format(cid=cid)), set())
    if not ids_raw:
        return []
    appts: list[dict[str, Any]] = []
    for raw_id in ids_raw:
        aid = raw_id if isinstance(raw_id, str) else raw_id.decode()
        a = await _get_service_appt(cid, aid)
        if a is not None:
            appts.append(a)
    return sorted(appts, key=lambda a: (a.get("date", ""), a.get("time", "")))


# ---------------------------------------------------------------------------
# Save / load service session state
# ---------------------------------------------------------------------------

async def save_service_session(session_id: str, state: dict[str, Any]) -> None:
    r = _get_client()
    if r is None:
        return
    key = _SESSION_KEY.format(session_id=session_id)
    await _safe(r.set(key, json.dumps(state), ex=3600))


async def load_service_session(session_id: str) -> dict[str, Any] | None:
    r = _get_client()
    if r is None:
        return None
    key = _SESSION_KEY.format(session_id=session_id)
    raw = await _safe(r.get(key))
    if raw is None:
        return None
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Book
# ---------------------------------------------------------------------------

_HELP_TYPES = (
    "a free service, paid service, periodic maintenance, body repair, "
    "accidental repair, warranty claim, or a general service"
)

SERVICE_HOURS_OPEN = 9
SERVICE_HOURS_CLOSE = 18


async def book_service(
    cid: str,
    service_type: str | None,
    date_str: str | None,
    hour: int | None,
    minute: int | None,
    vehicle_reg_no: str | None = None,
    notes: str | None = None,
    center: str = "default",
    date_error: str | None = None,
    time_error: str | None = None,
) -> ServiceResult:
    """Book a new service appointment (with multi-turn slot filling)."""

    if not service_type:
        return ServiceResult(
            f"What type of service do you need — {_HELP_TYPES}? "
            "Please also mention the preferred date and time.",
            pending={"type": "BOOK_SERVICE", "turns_left": PENDING_TTL_TURNS},
        )

    label = SERVICE_TYPE_LABEL.get(service_type, "service").lower()

    if date_error:
        return ServiceResult(
            f"I can't book that: {date_error}. Which date works for you?",
            pending={
                "type": "BOOK_SERVICE", "service_type": service_type,
                "vehicle_reg_no": vehicle_reg_no, "notes": notes,
                "center": center, "turns_left": PENDING_TTL_TURNS,
                **({"hour": hour, "minute": minute} if hour is not None else {}),
            },
        )

    if time_error:
        return ServiceResult(
            f"I can't book that: {time_error}. "
            f"What time between {SERVICE_HOURS_OPEN:02d}:00 and {SERVICE_HOURS_CLOSE:02d}:00 suits you?",
            pending={
                "type": "BOOK_SERVICE", "service_type": service_type,
                "vehicle_reg_no": vehicle_reg_no, "notes": notes,
                "center": center, "turns_left": PENDING_TTL_TURNS,
                **({"date": date_str} if date_str else {}),
            },
        )

    if date_str is None and hour is None:
        return ServiceResult(
            f"Happy to book your {label}. "
            f"Which date and time ({SERVICE_HOURS_OPEN:02d}:00–{SERVICE_HOURS_CLOSE:02d}:00) would you like?",
            pending={
                "type": "BOOK_SERVICE", "service_type": service_type,
                "vehicle_reg_no": vehicle_reg_no, "notes": notes,
                "center": center, "turns_left": PENDING_TTL_TURNS,
            },
        )

    if date_str is None:
        return ServiceResult(
            f"Which date would you like for your {label}?",
            pending={
                "type": "BOOK_SERVICE", "service_type": service_type,
                "hour": hour, "minute": minute or 0,
                "vehicle_reg_no": vehicle_reg_no, "notes": notes,
                "center": center, "turns_left": PENDING_TTL_TURNS,
            },
        )

    if hour is None:
        return ServiceResult(
            f"What time on {date_str} suits you for your {label}? "
            f"(e.g. 10:30, between {SERVICE_HOURS_OPEN:02d}:00 and {SERVICE_HOURS_CLOSE:02d}:00)",
            pending={
                "type": "BOOK_SERVICE", "service_type": service_type,
                "date": date_str, "vehicle_reg_no": vehicle_reg_no,
                "notes": notes, "center": center, "turns_left": PENDING_TTL_TURNS,
            },
        )

    # All slots filled — check availability and book
    appt_date = date.fromisoformat(date_str)
    time_str = f"{hour:02d}:{minute or 0:02d}"

    # Check slot availability
    ok = await slot_store.claim_slot(center, appt_date, time_str, f"pending-{cid}")
    if not ok:
        # Show available alternatives
        available = await slot_store.get_available(center, appt_date)
        if available:
            alt_times = ", ".join(s.time for s in available[:6])
            return ServiceResult(
                f"Sorry, the {time_str} slot on {date_str} is fully booked. "
                f"Available times that day: {alt_times}. Which would you prefer?"
            )
        return ServiceResult(
            f"Sorry, there are no available slots on {date_str}. "
            "Would you like to try a different date?"
        )

    appt_id = f"SVC-{uuid.uuid4().hex[:8].upper()}"
    appt: dict[str, Any] = {
        "service_appt_id": appt_id,
        "customer_id": cid,
        "service_type": service_type,
        "date": date_str,
        "time": time_str,
        "vehicle_reg_no": vehicle_reg_no,
        "notes": notes,
        "center": center,
        "status": ServiceAppointmentStatus.CONFIRMED.value,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
    }

    try:
        await _save_service_appt(cid, appt)
    except Exception:
        await slot_store.release_slot(center, appt_date, time_str)
        raise

    persistence.enqueue("service_appointment", f"{cid}:{appt_id}", appt)
    log.info("service appointment booked: cid=%s appt_id=%s type=%s date=%s time=%s",
             cid, appt_id, service_type, date_str, time_str)

    label_nice = SERVICE_TYPE_LABEL.get(service_type, "Service")
    text = (
        f"Booked ✓ {label_nice} on {date_str} at {time_str}. "
        f"Service Appointment ID: {appt_id}."
    )
    if vehicle_reg_no:
        text += f" Vehicle: {vehicle_reg_no}."

    # --- Engaging follow-up suggestions ---
    text += (
        "\n\n💡 **What else can I do for you?**"
        "\n• **Reschedule** or **cancel** this appointment if plans change"
        "\n• Check **available service slots** for another date"
        "\n• View your **service history**"
        "\nJust ask!"
    )

    return ServiceResult(
        text,
        {"type": "BOOK_SERVICE", "action_type": "BOOK_SERVICE", "result": appt},
        mutated=True, clear_pending=True,
    )


# ---------------------------------------------------------------------------
# Reschedule
# ---------------------------------------------------------------------------

async def reschedule_service(
    cid: str, appt_id: str | None,
    new_date_str: str | None = None, new_hour: int | None = None, new_minute: int | None = None,
    center: str | None = None,
) -> ServiceResult:
    """Reschedule an existing service appointment."""
    if not appt_id:
        appts = await list_service_appts(cid)
        active = [a for a in appts if a.get("status") == "confirmed"]
        if not active:
            return ServiceResult("You don't have any active service appointments to reschedule.")
        if len(active) == 1:
            appt_id = active[0]["service_appt_id"]
        else:
            lines = "\n".join(
                f"- {SERVICE_TYPE_LABEL.get(a['service_type'], 'Service')} on {a['date']} at {a['time']} "
                f"(ID: {a['service_appt_id']})"
                for a in active[:6]
            )
            return ServiceResult(
                f"Which service appointment should I reschedule?\n{lines}\n"
                "Please reply with the appointment ID.",
                pending={"type": "RESCHEDULE_SERVICE", "turns_left": PENDING_TTL_TURNS},
            )

    appt = await _get_service_appt(cid, appt_id)
    if appt is None:
        return ServiceResult(f"I couldn't find service appointment {appt_id} on your account.")
    if appt.get("status") != "confirmed":
        return ServiceResult(f"Service appointment {appt_id} is already {appt.get('status')}.")

    if new_date_str is None or new_hour is None:
        missing = []
        if new_date_str is None:
            missing.append("new date")
        if new_hour is None:
            missing.append("new time")
        return ServiceResult(
            f"When would you like to reschedule {appt_id} to? I need the {' and '.join(missing)}.",
            pending={
                "type": "RESCHEDULE_SERVICE", "service_appt_id": appt_id,
                "turns_left": PENDING_TTL_TURNS,
                **({"date": new_date_str} if new_date_str else {}),
                **({"hour": new_hour, "minute": new_minute or 0} if new_hour is not None else {}),
            },
        )

    new_date = date.fromisoformat(new_date_str)
    new_time = f"{new_hour:02d}:{new_minute or 0:02d}"
    use_center = center or appt.get("center", "default")

    # Release old slot
    old_date = date.fromisoformat(appt["date"])
    await slot_store.release_slot(appt.get("center", "default"), old_date, appt["time"])

    # Claim new slot
    ok = await slot_store.claim_slot(use_center, new_date, new_time, appt_id)
    if not ok:
        # Re-claim old slot (best-effort)
        await slot_store.claim_slot(appt.get("center", "default"), old_date, appt["time"], appt_id)
        return ServiceResult(
            f"Sorry, the {new_time} slot on {new_date_str} is not available. "
            "Please try a different date or time."
        )

    # Update appointment
    appt["date"] = new_date_str
    appt["time"] = new_time
    appt["center"] = use_center
    appt["status"] = ServiceAppointmentStatus.CONFIRMED.value
    appt["rescheduled_at"] = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    await _save_service_appt(cid, appt)

    persistence.enqueue("service_appointment", f"{cid}:{appt_id}", appt)
    log.info("service appointment rescheduled: cid=%s appt_id=%s new_date=%s new_time=%s",
             cid, appt_id, new_date_str, new_time)

    label = SERVICE_TYPE_LABEL.get(appt.get("service_type", ""), "Service")
    text = (
        f"Rescheduled ✓ {label} moved to {new_date_str} at {new_time}. ID: {appt_id}."
        "\n\n💡 Need anything else? I can **cancel** this appointment, "
        "**check other available slots**, or show your **service history**."
    )
    return ServiceResult(
        text,
        {"type": "RESCHEDULE_SERVICE", "action_type": "RESCHEDULE_SERVICE", "result": appt},
        mutated=True, clear_pending=True,
    )


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------

async def cancel_service(cid: str, appt_id: str | None, confirmed: bool = False) -> ServiceResult:
    """Cancel a service appointment (with confirmation step)."""
    if not appt_id:
        appts = await list_service_appts(cid)
        active = [a for a in appts if a.get("status") == "confirmed"]
        if not active:
            return ServiceResult("You don't have any active service appointments to cancel.")
        if len(active) == 1:
            appt_id = active[0]["service_appt_id"]
        else:
            lines = "\n".join(
                f"- {SERVICE_TYPE_LABEL.get(a['service_type'], 'Service')} on {a['date']} at {a['time']} "
                f"(ID: {a['service_appt_id']})"
                for a in active[:6]
            )
            return ServiceResult(
                f"Which service appointment should I cancel?\n{lines}\n"
                "Please reply with the appointment ID.",
            )

    appt = await _get_service_appt(cid, appt_id)
    if appt is None:
        return ServiceResult(f"I couldn't find service appointment {appt_id} on your account.")
    if appt.get("status") != "confirmed":
        return ServiceResult(f"Service appointment {appt_id} is already {appt.get('status')}.")

    if not confirmed:
        label = SERVICE_TYPE_LABEL.get(appt.get("service_type", ""), "Service")
        return ServiceResult(
            f"Just to confirm — cancel {label} on {appt['date']} at {appt['time']} "
            f"(ID: {appt_id})? Reply yes to confirm or no to keep it.",
            pending={
                "type": "CANCEL_SERVICE", "service_appt_id": appt_id,
                "turns_left": PENDING_TTL_TURNS,
            },
        )

    # Execute cancellation
    appt_date = date.fromisoformat(appt["date"])
    await slot_store.release_slot(appt.get("center", "default"), appt_date, appt["time"])
    appt["status"] = ServiceAppointmentStatus.CANCELLED.value
    appt["cancelled_at"] = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    await _save_service_appt(cid, appt)

    persistence.enqueue("service_appointment", f"{cid}:{appt_id}", appt)
    log.info("service appointment cancelled: cid=%s appt_id=%s", cid, appt_id)

    label = SERVICE_TYPE_LABEL.get(appt.get("service_type", ""), "Service")
    text = (
        f"Cancelled ✓ {label} on {appt['date']} at {appt['time']} (ID: {appt_id})."
        "\n\n💡 Would you like to **book a new service appointment** instead, "
        "or is there anything else I can help with?"
    )
    return ServiceResult(
        text,
        {"type": "CANCEL_SERVICE", "action_type": "CANCEL_SERVICE", "result": appt},
        mutated=True, clear_pending=True,
    )


# ---------------------------------------------------------------------------
# Check slots (read-only)
# ---------------------------------------------------------------------------

async def check_available_slots(center: str, dt: date) -> ServiceResult:
    """Return a formatted list of available service slots for a date."""
    available = await slot_store.get_available(center, dt)
    if not available:
        return ServiceResult(
            f"No available service slots on {dt.isoformat()}. "
            "Would you like to check a different date?"
        )
    lines = "\n".join(
        f"- {s.time} ({s.available} spot{'s' if s.available != 1 else ''} available)"
        for s in available
    )
    return ServiceResult(
        f"Available service slots on {dt.isoformat()} at {center} center:\n{lines}\n"
        "Which time would you like to book?"
    )


# ---------------------------------------------------------------------------
# Service history (read-only)
# ---------------------------------------------------------------------------

async def service_history(cid: str) -> ServiceResult:
    """Show all service appointments for a customer."""
    appts = await list_service_appts(cid)
    if not appts:
        return ServiceResult("You don't have any service appointments on record.")
    lines = []
    for a in appts[:10]:
        label = SERVICE_TYPE_LABEL.get(a.get("service_type", ""), "Service")
        status = a.get("status", "unknown")
        lines.append(f"- {label} on {a['date']} at {a['time']} — {status} (ID: {a['service_appt_id']})")
    more = f"\n…and {len(appts) - 10} more." if len(appts) > 10 else ""
    return ServiceResult(f"Your service appointments:\n" + "\n".join(lines) + more)
