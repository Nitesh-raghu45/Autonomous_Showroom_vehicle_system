"""Slot extraction for the Service Agent.

Parses user utterances to extract structured slot values (service_type, date,
time, vehicle registration, etc.) using the same deterministic approach as the
Sales Agent's ``router.when`` and ``router.extract``.
"""
from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any

from ... import clock
from ...router.when import When, parse_when as _parse_when


# ---------------------------------------------------------------------------
# Service-type extraction
# ---------------------------------------------------------------------------

_SERVICE_TYPE_MAP: dict[str, str] = {
    # free service variants
    "free service": "FREE_SERVICE",
    "complimentary service": "FREE_SERVICE",
    "first service": "FREE_SERVICE",
    "1st service": "FREE_SERVICE",
    "second service": "FREE_SERVICE",
    "2nd service": "FREE_SERVICE",
    "third service": "FREE_SERVICE",
    "3rd service": "FREE_SERVICE",
    # paid service
    "paid service": "PAID_SERVICE",
    "regular service": "PAID_SERVICE",
    # periodic maintenance
    "periodic maintenance": "PERIODIC_MAINTENANCE",
    "periodic service": "PERIODIC_MAINTENANCE",
    "routine maintenance": "PERIODIC_MAINTENANCE",
    "scheduled maintenance": "PERIODIC_MAINTENANCE",
    "routine service": "PERIODIC_MAINTENANCE",
    # body repair
    "body repair": "BODY_REPAIR",
    "dent repair": "BODY_REPAIR",
    "scratch repair": "BODY_REPAIR",
    "paint job": "BODY_REPAIR",
    "denting painting": "BODY_REPAIR",
    "denting and painting": "BODY_REPAIR",
    # accidental repair
    "accident repair": "ACCIDENTAL_REPAIR",
    "accidental repair": "ACCIDENTAL_REPAIR",
    "collision repair": "ACCIDENTAL_REPAIR",
    "crash repair": "ACCIDENTAL_REPAIR",
    # warranty
    "warranty claim": "WARRANTY_CLAIM",
    "warranty service": "WARRANTY_CLAIM",
    "warranty repair": "WARRANTY_CLAIM",
    "under warranty": "WARRANTY_CLAIM",
    # general catch-all
    "general service": "GENERAL",
    "car service": "GENERAL",
    "vehicle service": "GENERAL",
    "service": "GENERAL",
    "oil change": "GENERAL",
    "tyre rotation": "GENERAL",
    "tire rotation": "GENERAL",
    "wheel alignment": "GENERAL",
    "brake check": "GENERAL",
    "battery check": "GENERAL",
    "ac service": "GENERAL",
    "ac repair": "GENERAL",
    "air conditioning": "GENERAL",
}

# Sorted longest-first so "periodic maintenance" matches before "service"
_SERVICE_TYPE_PATTERNS = sorted(_SERVICE_TYPE_MAP.keys(), key=len, reverse=True)


def extract_service_type(text: str) -> str | None:
    """Extract service type from user text. Returns the enum value or None."""
    lower = text.lower()
    for pattern in _SERVICE_TYPE_PATTERNS:
        if pattern in lower:
            return _SERVICE_TYPE_MAP[pattern]
    return None


# ---------------------------------------------------------------------------
# Vehicle registration number extraction
# ---------------------------------------------------------------------------

_VEHICLE_REG_RE = re.compile(
    r"\b([A-Z]{2}\s*\d{1,2}\s*[A-Z]{1,3}\s*\d{4})\b", re.IGNORECASE
)


def extract_vehicle_reg(text: str) -> str | None:
    """Extract an Indian-style vehicle registration number (e.g. MH12AB1234)."""
    m = _VEHICLE_REG_RE.search(text.upper())
    return re.sub(r"\s+", "", m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Service appointment ID extraction
# ---------------------------------------------------------------------------

_SERVICE_APPT_ID_RE = re.compile(r"\b(SVC-[A-Z0-9]{8,})\b", re.IGNORECASE)


def extract_service_appt_id(text: str) -> str | None:
    """Extract a service appointment ID like SVC-XXXXXXXX."""
    m = _SERVICE_APPT_ID_RE.search(text.upper())
    return m.group(1).upper() if m else None


# ---------------------------------------------------------------------------
# Composite slot extraction
# ---------------------------------------------------------------------------

def extract_all_slots(text: str, pending: dict[str, Any] | None = None) -> dict[str, Any]:
    """Extract all service-related slots from the user's message.

    Merges with any ``pending`` context from a previous turn so that
    multi-turn slot filling works naturally.
    """
    pending = dict(pending or {})
    slots: dict[str, Any] = {}

    # Service type
    stype = extract_service_type(text)
    if stype:
        slots["service_type"] = stype
    elif pending.get("service_type"):
        slots["service_type"] = pending["service_type"]

    # Date / time (reuse the sales agent's When parser — it's a pure function)
    when: When = _parse_when(text, clock.today())
    if when.date is not None:
        slots["date"] = when.date.isoformat()
    elif pending.get("date"):
        slots["date"] = pending["date"]

    if when.hour is not None:
        slots["hour"] = when.hour
        slots["minute"] = when.minute or 0
    elif pending.get("hour") is not None:
        slots["hour"] = pending["hour"]
        slots["minute"] = pending.get("minute", 0)

    if when.date_error:
        slots["date_error"] = when.date_error
    if when.time_error:
        slots["time_error"] = when.time_error

    # Vehicle registration
    vreg = extract_vehicle_reg(text)
    if vreg:
        slots["vehicle_reg_no"] = vreg
    elif pending.get("vehicle_reg_no"):
        slots["vehicle_reg_no"] = pending["vehicle_reg_no"]

    # Service appointment ID (for reschedule / cancel)
    appt_id = extract_service_appt_id(text)
    if appt_id:
        slots["service_appt_id"] = appt_id
    elif pending.get("service_appt_id"):
        slots["service_appt_id"] = pending["service_appt_id"]

    # Notes — anything in quotes or after "notes:" / "comment:"
    notes_match = re.search(r'(?:notes?|comments?|reason)\s*[:=]\s*["\']?(.+?)["\']?\s*$', text, re.IGNORECASE)
    if notes_match:
        slots["notes"] = notes_match.group(1).strip()

    # Center
    if pending.get("center"):
        slots["center"] = pending["center"]

    return slots
