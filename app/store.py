"""In-process seed data + profile rendering.

Redis is the primary mutable store (see cache.py). This module:
  1. Loads the JSON file once at startup as the seed dataset
  2. Seeds Redis on first boot (skips if keys already exist)
  3. Serves as the read-only fallback when Redis is unreachable
  4. Renders a customer record into the compact text block used in LLM prompts
     (only on the LLM path - deterministic answers never render prose).
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger("store")

_RECORDS: list[dict[str, Any]] = []
_BY_ID: dict[str, dict[str, Any]] = {}
_BY_MOBILE: dict[str, str] = {}


def load(path: Path) -> int:
    """Load records from disk. Returns how many were indexed."""
    global _RECORDS
    with open(path, "r", encoding="utf-8") as fh:
        _RECORDS = json.load(fh)

    _BY_ID.clear()
    _BY_MOBILE.clear()
    for rec in _RECORDS:
        cust = rec.get("customer") or {}
        cid = str(cust.get("customer_id", "")).strip()
        if not cid:
            continue
        _BY_ID[cid] = rec
        for key in ("mobile_no", "alt_mobile_no"):
            num = cust.get(key)
            if num:
                _BY_MOBILE[_normalise_mobile(str(num))] = cid
    render_profile.cache_clear()
    return len(_BY_ID)


async def seed_redis() -> None:
    """Seed Redis with all JSON records on first boot (single pipelined round trip)."""
    from . import cache
    await cache.seed_customers(_BY_ID, _normalise_mobile)


def _normalise_mobile(num: str) -> str:
    digits = "".join(ch for ch in num if ch.isdigit())
    return digits[-10:] if len(digits) > 10 else digits


normalise_mobile = _normalise_mobile


def resolve_id(user_input: str) -> str | None:
    """Synchronous fallback — checks in-memory JSON index only."""
    raw = (user_input or "").strip()
    if raw in _BY_ID:
        return raw
    return _BY_MOBILE.get(_normalise_mobile(raw))


def get(customer_id: str) -> dict[str, Any] | None:
    return _BY_ID.get(str(customer_id).strip())


def all_ids() -> list[str]:
    return sorted(_BY_ID)


def summaries() -> list[dict[str, str]]:
    out = []
    for cid in all_ids():
        c = _BY_ID[cid]["customer"]
        e = _BY_ID[cid].get("enquiry") or {}
        out.append({
            "customer_id":     cid,
            "name":            c.get("name", ""),
            "mobile_no":       c.get("mobile_no", ""),
            "enquiry_status":  e.get("status", ""),
            "vehicle":         f"{e.get('interested_make', '')} {e.get('interested_model', '')}".strip(),
        })
    return out


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def _date(value: Any) -> str:
    if not value:
        return "not set"
    text = str(value)
    return text.split("T")[0] if "T" in text else text


def _money(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"INR {float(value):,.2f}"
    except (TypeError, ValueError):
        return str(value)


def _lines(pairs: Iterable[tuple[str, Any]]) -> list[str]:
    return [f"- {label}: {'not on record' if v in (None, '', []) else v}" for label, v in pairs]


def render_record(record: dict[str, Any], customer_id: str) -> str:
    """Render a (possibly Redis-sourced) record dict as a markdown profile block."""
    c = record.get("customer") or {}
    e = record.get("enquiry") or {}
    q = record.get("quotation")
    sched = record.get("scheduled_contact")
    logs = record.get("contact_log") or []

    out: list[str] = [f"### CUSTOMER RECORD {customer_id}", "", "**Customer**"]
    out += _lines([
        ("Name",             f"{c.get('title', '')} {c.get('name', '')}".strip()),
        ("Customer ID",      c.get("customer_id")),
        ("Mobile",           c.get("mobile_no")),
        ("Alternate mobile", c.get("alt_mobile_no")),
        ("Email",            c.get("email")),
    ])

    out += ["", "**Enquiry**"]
    out += _lines([
        ("Enquiry ID",        e.get("enquiry_id")),
        ("Status",            e.get("status")),
        ("Raised on",         _date(e.get("created_at"))),
        ("Salesperson",       e.get("salesperson")),
        ("New or used",       e.get("new_or_used")),
        ("Interested vehicle", f"{e.get('interested_make', '')} {e.get('interested_model', '')}".strip()),
    ])

    out += ["", "**Quotation / proposal**"]
    if not q:
        out.append("- No quotation has been raised for this customer.")
    else:
        veh = q.get("vehicle") or {}
        out += _lines([
            ("Proposal ID",     q.get("proposal_id")),
            ("Proposal status", q.get("status")),
            ("Delivery date",   _date(q.get("delivery_date"))),
            ("Vehicle",         f"{veh.get('make', '')} {veh.get('model', '')}".strip()),
            ("Stock ID",        veh.get("stock_id")),
        ])
        breakdown = q.get("cost_breakdown") or []
        if breakdown:
            sale = sum(float(i.get("sale_amount") or 0) for i in breakdown)
            vat  = sum(float(i.get("vat_amount")  or 0) for i in breakdown)
            out.append("- Cost breakdown:")
            for item in breakdown:
                out.append(
                    f"  - {item.get('description', 'item')}: sale {_money(item.get('sale_amount'))}, "
                    f"tax {_money(item.get('vat_amount'))}"
                )
            out.append(f"- Total sale amount: {_money(sale)}")
            out.append(f"- Total tax: {_money(vat)}")
            out.append(f"- Grand total payable: {_money(sale + vat)}")

    out += ["", "**Scheduled contact**"]
    if not sched:
        out.append("- Nothing scheduled.")
    else:
        hh, mm = sched.get("sched_hour"), sched.get("sched_minute")
        time_str = f"{hh:02d}:{mm:02d}" if isinstance(hh, int) and isinstance(mm, int) else "time not set"
        out += _lines([
            ("Date",   _date(sched.get("sched_date"))),
            ("Time",   time_str),
            ("Type",   sched.get("contact_type")),
            ("Status", sched.get("status")),
        ])

    out += ["", f"**Contact history ({len(logs)} entr" + ("y" if len(logs) == 1 else "ies") + ")**"]
    if not logs:
        out.append("- No contact has been logged.")
    else:
        for entry in sorted(logs, key=lambda x: str(x.get("log_date") or "")):
            out.append(
                f"- {_date(entry.get('log_date'))} | {entry.get('contact_type', 'CONTACT')} | "
                f"{entry.get('notes', 'no notes')}"
            )

    return "\n".join(out)


@lru_cache(maxsize=512)
def render_profile(customer_id: str) -> str | None:
    """Render a JSON-sourced record. Used when Redis has no mutable override."""
    rec = get(customer_id)
    if rec is None:
        return None
    return render_record(rec, customer_id)


def render_appointments(appts: list[dict[str, Any]]) -> str:
    """Compact text for appointments booked through the assistant (LLM context only)."""
    if not appts:
        return ""
    lines = ["**Booked appointments**"]
    for a in appts:
        lines.append(
            f"- {a.get('appointment_type')} on {a.get('date')} at {a.get('time')} "
            f"[{a.get('status')}] (ID {a.get('appointment_id')})"
            + (f" - {a['notes']}" if a.get("notes") else "")
        )
    return "\n".join(lines)
