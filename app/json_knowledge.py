"""JSON-backed knowledge context for the LLM fallback.

The rule router remains the first choice.  This module is only used after the
router defers.  It turns the authoritative application data into a compact,
explicit DATA block so the LLM can answer questions whose answer exists in JSON
even when no hand-written rule exists.
"""
from __future__ import annotations

import json
import re
from typing import Any

from . import inventory

_INV_WORDS = re.compile(
    r"\b(car|cars|vehicle|vehicles|inventory|stock|available|price|cost|"
    r"model|make|brand|variant|fuel|diesel|petrol|electric|ev|hybrid|cng|"
    r"automatic|manual|suv|sedan|hatchback|mpv|seat|mileage|km|cheapest|"
    r"expensive|color|colour|year|compare)\b", re.I
)

def _compact(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)

# Contact identifiers leave the process (cloud LLM) only when the question is about them.
_CONTACT_WORDS = re.compile(
    r"\b(e-?mail|mail|mobile|phone|number|contact|alt\w*|secondary|details|profile|info\w*|account|record)\b", re.I
)
_CONTACT_FIELDS = ("mobile_no", "alt_mobile_no", "email")


def _mask(field: str, value: Any) -> Any:
    if not isinstance(value, str) or not value:
        return value
    if field == "email":
        local, _, domain = value.partition("@")
        return f"{local[:1]}***@{domain}" if domain else "***"
    digits = re.sub(r"\D", "", value)
    return f"******{digits[-4:]}" if len(digits) >= 4 else "***"


def _redact_record(record: dict[str, Any], question: str) -> dict[str, Any]:
    if _CONTACT_WORDS.search(question or ""):
        return record
    cust = record.get("customer")
    if not isinstance(cust, dict):
        return record
    masked = {k: (_mask(k, v) if k in _CONTACT_FIELDS else v) for k, v in cust.items()}
    return {**record, "customer": masked}


def _customer_context(record: dict[str, Any] | None, cid: str | None, question: str = "") -> str:
    if not record or not cid:
        return ""
    # The authoritative customer JSON, with contact identifiers masked unless the question needs them.
    return "CUSTOMER_JSON\n" + _compact(_redact_record(record, question))

_ANAPHOR = re.compile(r"\b(it|its|it's|that|this|those|these|them|they|the same|one|ones)\b", re.I)


def _inventory_context(question: str, limit: int = 24, focus: list[dict[str, Any]] | None = None) -> str:
    # Use the existing deterministic extractor when possible.  If it cannot
    # extract filters, return a small catalogue sample rather than inventing.
    try:
        from .router import extract as X
        f = X.inventory_filters(question)
    except Exception:
        f = {}

    # Follow-up about the vehicles already under discussion ("and its price?"):
    # ground on those rows rather than a fresh, context-blind catalogue search.
    if focus and not (f.get("makes") or f.get("models")) and (_INV_WORDS.search(question) or _ANAPHOR.search(question)):
        return "INVENTORY_JSON (vehicles currently being discussed)\n" + _compact(focus[:limit])

    if not _INV_WORDS.search(question):
        return ""
    cat = inventory.catalog()

    kwargs = {
        "makes": f.get("makes"),
        "models": f.get("models"),
        "vtype": f.get("vtype"),
        "fuel": f.get("fuel") or f.get("fuels"),
        "transmission": f.get("transmission"),
        "body": f.get("body"),
        "min_price": f.get("min_price"),
        "max_price": f.get("max_price"),
        "max_km": f.get("max_km"),
        "min_seats": f.get("min_seats"),
        "max_seats": f.get("max_seats"),
        "years": f.get("years"),
        "min_year": f.get("min_year"),
        "colors": f.get("colors"),
        "variants": f.get("variants"),
        "sort": f.get("sort"),
        "limit": limit,
    }
    _, rows = cat.search(**kwargs)

    # For broad catalogue questions, include the complete small catalogue.
    # The current catalogue is intentionally small; this keeps the answer
    # grounded in the same source used by deterministic inventory actions.
    if not rows and not f:
        rows = cat.items[:limit]

    return "INVENTORY_JSON\n" + _compact(rows)

def build_json_context(
    *,
    question: str,
    cid: str | None,
    record: dict[str, Any] | None,
    appts: list[dict[str, Any]],
    focus: list[dict[str, Any]] | None = None,
) -> str:
    parts = []
    customer = _customer_context(record, cid, question)
    if customer:
        parts.append(customer)

    if appts:
        parts.append("BOOKED_APPOINTMENTS_JSON\n" + _compact(appts))

    inv = _inventory_context(question, focus=focus)
    if inv:
        parts.append(inv)

    return "\n\n".join(parts) or "JSON_CONTEXT\n{}"
