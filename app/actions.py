"""Validated business actions, shared by the deterministic router AND the LLM planner.

Whoever decides *what* to do (rules or model), *doing* it always goes through here:
pydantic validation, business-hour checks, atomic writes, and templated confirmations.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from pydantic import ValidationError

from . import cache, clock, inventory, inventory_query, inventory_renderer, mutations, render
from .config import settings
from .models import AppointmentType, CustomerRegisterRequest, CustomerUpdateRequest, ScheduleRequest
from .router.when import BUSINESS_CLOSE, BUSINESS_OPEN, When, fmt_time, hours_range

log = logging.getLogger("actions")

PENDING_TTL_TURNS = 3


@dataclass
class Result:
    text: str
    action: dict[str, Any] | None = None
    mutated: bool = False
    pending: dict[str, Any] | None = None       # new pending confirmation to store
    clear_pending: bool = False
    ctx: dict[str, Any] = field(default_factory=dict)


def _pyd_msg(exc: ValidationError) -> str:
    return "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg'].replace('Value error, ', '')}" for e in exc.errors())


# ---------------------------------------------------------------------------
# booking
# ---------------------------------------------------------------------------

_HELP_TYPES = "a test drive, callback, advance payment, document submission or a general appointment"


async def book(cid: str, appt_type: str | None, when: When, notes: str | None, appts: list[dict[str, Any]] | None = None, browsed_vehicles: list[dict[str, Any]] | None = None) -> Result:
    if not appt_type:
        return Result(f"What would you like to book - {_HELP_TYPES}? Please include the date and time too.")

    # --- Multi-car disambiguation ---
    # If the user is booking a test drive (or any appointment) without specifying a vehicle,
    # and they have browsed multiple cars in this session, ask which car they want.
    if appt_type == "TEST_DRIVE" and not notes and browsed_vehicles and len(browsed_vehicles) > 1:
        lines = []
        for v in browsed_vehicles[:6]:
            lines.append(f"- {v.get('make', '')} {v.get('model', '')} ({v.get('stock_id', '')}){' — ' + render.lakh(v['price']) if v.get('price') else ''}")
        return Result(
            f"You've been looking at {len(browsed_vehicles)} vehicles. Which one would you like to test drive?\n"
            + "\n".join(lines)
            + "\n\nJust mention the car name or stock ID, along with your preferred date and time.",
            pending={"type": "BOOK_APPOINTMENT", "appointment_type": appt_type, "turns_left": PENDING_TTL_TURNS},
        )

    # Answer in the customer's own clock: "10 PM" in -> "10 PM" out, "22:00" in -> "22:00" out.
    ampm = when.ampm
    hrs = hours_range(ampm, " and ")
    base_pending: dict[str, Any] = {"type": "BOOK_APPOINTMENT", "appointment_type": appt_type, "notes": notes,
                                    "turns_left": PENDING_TTL_TURNS, "ampm": ampm}
    if when.date_error:
        pending_err: dict[str, Any] = dict(base_pending)
        # Preserve the valid hour/minute even though the date was bad
        if when.hour is not None and not when.time_error:
            pending_err["hour"] = when.hour
            pending_err["minute"] = when.minute
        return Result(f"I can't book that: {when.date_error}. Which date works for you (from today onwards)?", pending=pending_err)
    if when.time_error:
        pending_err2: dict[str, Any] = dict(base_pending)
        # Preserve the valid date even though the time was bad
        if when.date is not None:
            pending_err2["date"] = when.date.isoformat()
        on_day = f" on {when.date.strftime('%A, %d %B')}" if when.date is not None else ""
        return Result(f"I can't book that: {when.time_error}. What time{on_day} between {hrs} suits you?", pending=pending_err2)
    label = render.TYPE_LABEL.get(appt_type, "appointment").lower()
    if when.date is None and when.hour is None:
        return Result(
            f"Happy to book your {label}. Which date and exact time (between {hrs}) would you like?",
            pending=dict(base_pending),
        )
    if when.date is None:
        pending_data: dict[str, Any] = dict(base_pending)
        # Preserve the valid hour/minute so the next slot-fill turn doesn't lose it
        if when.hour is not None and not when.time_error:
            pending_data["hour"] = when.hour
            pending_data["minute"] = when.minute
        at = f" at {fmt_time(when.hour, when.minute, ampm)}" if when.hour is not None else ""
        return Result(f"Which date would you like for your {label}{at}?", pending=pending_data)
    if when.hour is None:
        example = "10:30 AM" if ampm else "10:30"
        return Result(
            f"What exact time on {when.date.strftime('%A, %d %B')} suits you for your {label}? (e.g. {example}, between {hrs})",
            pending={**base_pending, "date": when.date.isoformat()},
        )
    if when.date == clock.today() and not clock.is_frozen():
        n = clock.now()
        if (when.hour, when.minute) <= (n.hour, n.minute):
            return Result(f"{fmt_time(when.hour, when.minute, ampm)} today has already passed. Would you like a later time or another day?")
    slot_time = f"{when.hour:02d}:{when.minute:02d}"
    for a in appts or []:
        if a.get("status") == "confirmed" and a.get("date") == when.date.isoformat() and a.get("time") == slot_time:
            same = a.get("appointment_type") == appt_type
            return Result(
                f"You already have {'this' if same else 'another'} appointment at that time: "
                f"{render.TYPE_LABEL.get(a['appointment_type'], 'Appointment')} on {a['date']} at {a['time']} (ID {a['appointment_id']}). "
                + ("No new booking was made." if same else "Please pick a different time."))
    try:
        req = ScheduleRequest(
            appointment_type=AppointmentType(appt_type), date=when.date, hour=when.hour,
            minute=when.minute, notes=notes,
        )
        appt = await mutations.book_appointment(cid, req)
    except ValidationError as exc:
        return Result(f"I couldn't book that: {_pyd_msg(exc)}.")
    except mutations.Conflict as exc:
        return Result(str(exc))
    except cache.StoreUnavailable:
        return Result("I couldn't save that booking right now. Please try again in a moment.")
    shown = fmt_time(when.hour, when.minute, ampm)
    note = f" (I read your time as {shown}.)" if when.time_assumed else ""
    text = (f"Booked ✓ {render.TYPE_LABEL[appt['appointment_type']]} on {appt['date']} "
            f"({when.date.strftime('%A')}) at {shown}. Appointment ID {appt['appointment_id']}.{note}")
    # --- Engaging follow-up suggestions ---
    followup = _booking_followup(appt_type, notes)
    if followup:
        text += followup
    return Result(text, {"type": "BOOK_APPOINTMENT", "action_type": "BOOK_APPOINTMENT", "result": appt}, mutated=True, clear_pending=True)


def _booking_followup(appt_type: str | None, notes: str | None) -> str:
    """Generate a contextual follow-up suggestion after a successful booking."""
    if appt_type == "TEST_DRIVE":
        return (
            "\n\n💡 **What's next?** I can also:"
            "\n• Get you a **price quotation** for this vehicle"
            "\n• Show you **similar vehicles** or compare options"
            "\n• Help with **financing** or **exchange** details"
            "\nJust let me know!"
        )
    if appt_type == "CALLBACK":
        return (
            "\n\n💡 Meanwhile, would you like to:"
            "\n• Browse our **inventory** for your preferred vehicle?"
            "\n• **Book a test drive** at the showroom?"
            "\n• Check your **quotation** details?"
        )
    if appt_type == "ADVANCE_PAYMENT":
        return (
            "\n\n💡 While you're here, would you like to:"
            "\n• Review your **quotation breakdown**?"
            "\n• Check the **delivery timeline**?"
        )
    if appt_type == "DOCUMENT_SUBMISSION":
        return (
            "\n\n💡 Need anything else? I can:"
            "\n• Check your **appointment schedule**"
            "\n• Show your **enquiry status**"
        )
    return (
        "\n\n💡 Anything else I can help with? I can assist with inventory browsing, "
        "quotations, or booking another appointment."
    )


# ---------------------------------------------------------------------------
# update info
# ---------------------------------------------------------------------------

_UPDATE_HELP = {
    "email_invalid": "That email address doesn't look valid (it needs the form name@domain.com).",
    "email_missing": "What email address should I change it to?",
    "name_missing": "What name should I change it to?",
    "alt_missing": "What is the 10-digit alternate mobile number?",
    "mobile_missing": "What is the new 10-digit primary mobile number?",
    "nothing_specified": "What would you like to update - your name, email, or alternate mobile number?",
}


async def update(cid: str, s: dict[str, Any]) -> Result:
    if s.get("ambiguous_number"):
        n = s["ambiguous_number"]
        if len(n) != 10:
            return Result("Mobile numbers must be exactly 10 digits. Please send the full number again.")
        return Result(f"Should {n} be your primary mobile number (the one you sign in with) or your alternate number? Reply 'primary' or 'alternate'.",
                      pending={"type": "UPDATE_NUMBER", "number": n, "turns_left": PENDING_TTL_TURNS})
    msgs = [_UPDATE_HELP[p] for p in s.get("problems", []) if p in _UPDATE_HELP]
    refused = s.get("refused") or []
    if refused:
        msgs.append("Your title can't be changed here - please contact the dealership.")
    fields = {k: s.get(k) for k in ("name", "email", "mobile_no", "alt_mobile_no") if s.get(k)}
    if not fields and not s.get("clear_alt"):
        return Result(" ".join(msgs) if msgs else _UPDATE_HELP["nothing_specified"])
    try:
        req = CustomerUpdateRequest(clear_alt_mobile=bool(s.get("clear_alt")), **fields)
    except ValidationError as exc:
        bad = {e["loc"][0] for e in exc.errors()}
        parts = []
        if "email" in bad:
            parts.append(_UPDATE_HELP["email_invalid"])
        if "alt_mobile_no" in bad:
            parts.append("The alternate number must be exactly 10 digits.")
        if "mobile_no" in bad:
            parts.append("The primary mobile number must be exactly 10 digits.")
        if "name" in bad:
            parts.append("A name needs at least 2 letters.")
        return Result(" ".join(parts + msgs) or _pyd_msg(exc))
    try:
        res = await mutations.update_customer(cid, req)
    except mutations.Conflict as exc:
        return Result(str(exc))
    except cache.StoreUnavailable:
        return Result("I couldn't save that change right now. Please try again in a moment.")
    changed = res["updated_fields"]
    if not changed:
        return Result(" ".join(["That's already what's on file - nothing to change."] + msgs))
    c = res["record"]["customer"]
    bits = []
    if "name" in changed:
        bits.append(f"name: {c['name']}")
    if "email" in changed:
        bits.append(f"email: {c['email']}")
    if "mobile_no" in changed:
        bits.append(f"primary mobile: {c['mobile_no']} (use it to sign in from now on)")
    if "alt_mobile_no" in changed:
        bits.append(f"alternate mobile: {c['alt_mobile_no']}" if c.get("alt_mobile_no") else "alternate mobile removed")
    text = "Done ✓ Updated - " + "; ".join(bits) + "." + (" " + " ".join(msgs) if msgs else "")
    return Result(text, {"type": "UPDATE_INFO", "action_type": "UPDATE_INFO", "result": res}, mutated=True, clear_pending=True)


async def select_vehicle(cid: str, stock_id: str | None) -> Result:
    wanted = (stock_id or "").replace("-", "").upper()
    row = next((r for r in inventory.catalog().items if str(r.get("stock_id", "")).replace("-", "").upper() == wanted), None)
    if row is None:
        return Result("I couldn't identify that vehicle. Please select it using its stock ID.")
    res = await mutations.select_vehicle(cid, row)
    return Result(
        f"Selected {row['make']} {row['model']} ({row['stock_id']}) for your enquiry.",
        {"type": "SELECT_VEHICLE", "action_type": "SELECT_VEHICLE", "result": res}, mutated=True,
    )


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------

async def register(s: dict[str, Any]) -> Result:
    missing = [lbl for key, lbl in (("name", "your full name"), ("mobile_no", "a 10-digit mobile number"), ("email", "an email address")) if not s.get(key)]
    probs = []
    if "email_invalid" in s.get("problems", []):
        probs.append("that email address doesn't look valid (it needs the form name@domain.com)")
        missing = [m for m in missing if m != "an email address"]
    if "mobile_invalid" in s.get("problems", []):
        probs.append("the mobile number must be exactly 10 digits")
        missing = [m for m in missing if m != "a 10-digit mobile number"]
    def _join(items: list[str]) -> str:
        return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]

    if missing or probs:
        parts = []
        if probs:
            parts.append("There's a problem: " + "; ".join(probs) + ".")
        if missing:
            parts.append("To register you I still need " + _join(missing) + ".")
        return Result(" ".join(parts))
    try:
        req = CustomerRegisterRequest(**{k: v for k, v in s.items() if k not in ("problems",) and v is not None})
    except ValidationError as exc:
        bad = {e["loc"][0] for e in exc.errors()}
        parts = []
        if "name" in bad:
            parts.append("a name needs at least 2 letters")
        if "email" in bad:
            parts.append("the email address isn't valid")
        if "mobile_no" in bad or "alt_mobile_no" in bad:
            parts.append("mobile numbers must be exactly 10 digits")
        return Result("I couldn't register you: " + ("; ".join(parts) or _pyd_msg(exc)) + ".")
    try:
        res = await mutations.register_customer(req)
    except mutations.Conflict as exc:
        return Result(str(exc))
    except cache.StoreUnavailable:
        return Result("I couldn't complete the registration right now. Please try again in a moment.")
    text = (
        f"Welcome, {req.name}! You're registered ✓ Your customer ID is {res['customer_id']} "
        f"and your enquiry ID is {res['enquiry_id']}."
        "\n\n🎉 **Great to have you on board!** Here's what I can do for you:"
        "\n• Browse our **vehicle inventory** — just say a brand or type"
        "\n• **Book a test drive** at the showroom"
        "\n• Check **pricing and quotations**"
        "\nWhat would you like to explore first?"
    )
    return Result(
        text,
        {"type": "REGISTER_CUSTOMER", "action_type": "REGISTER_CUSTOMER", "result": res}, mutated=True,
    )


# ---------------------------------------------------------------------------
# cancel (two-step: propose -> confirm)
# ---------------------------------------------------------------------------

def _active(appts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [a for a in appts if a.get("status") == "confirmed"]


def _line(a: dict[str, Any]) -> str:
    return f"{render.TYPE_LABEL.get(a['appointment_type'], 'Appointment')} on {a['date']} at {a['time']} (ID {a['appointment_id']})"


async def cancel_request(rec: dict[str, Any], appts: list[dict[str, Any]], s: dict[str, Any]) -> Result:
    want_id = s.get("appt_id")
    active = _active(appts)
    if want_id:
        match = [a for a in appts if a["appointment_id"].upper() == want_id.upper()]
        if not match:
            return Result(f"I couldn't find an appointment with ID {want_id} on your account.")
        if match[0].get("status") != "confirmed":
            return Result(f"Appointment {want_id} is already {match[0].get('status')}.")
        active = match
    elif s.get("appointment_type"):
        active = [a for a in active if a["appointment_type"] == s["appointment_type"]]
    if not active:
        sc = rec.get("scheduled_contact")
        if sc and sc.get("status") in ("pending", "confirmed") and not appts:
            return Result(f"Your {sc.get('contact_type') or 'scheduled contact'} on {render.d(sc.get('sched_date'))} was arranged by the dealership, so I can't cancel it here - please contact the showroom.")
        return Result("You don't have any matching active appointments to cancel.")
    if len(active) == 1 or s.get("all"):
        ids = [a["appointment_id"] for a in active]
        what = _line(active[0]) if len(active) == 1 else f"all {len(active)} appointments:\n" + "\n".join("- " + _line(a) for a in active[:6]) + (f"\n…and {len(active) - 6} more" if len(active) > 6 else "")
        return Result(f"Just to confirm - cancel {what}? Reply yes to confirm or no to keep it.",
                      pending={"type": "CANCEL_APPOINTMENT", "ids": ids, "turns_left": PENDING_TTL_TURNS})
    shown = active[:6]
    extra = f"\n…and {len(active) - 6} more." if len(active) > 6 else ""
    return Result("Which appointment should I cancel? Reply with its type or ID:\n" + "\n".join("- " + _line(a) for a in shown) + extra)


async def cancel_confirm(cid: str, pending: dict[str, Any]) -> Result:
    done = []
    try:
        for aid in pending.get("ids", []):
            done.append(await mutations.cancel_appointment(cid, aid))
    except KeyError as exc:
        return Result(str(exc), clear_pending=True)
    except cache.StoreUnavailable:
        return Result("I couldn't cancel that right now. Please try again in a moment.", clear_pending=True)
    text = "Cancelled ✓ " + "; ".join(_line(a) for a in done) + "."
    text += (
        "\n\n💡 Would you like to **book a new appointment** at a different time, "
        "or is there anything else I can help with?"
    )
    return Result(text, {"type": "CANCEL_APPOINTMENT", "action_type": "CANCEL_APPOINTMENT", "result": done[0] if len(done) == 1 else {"cancelled": done}},
                  mutated=True, clear_pending=True)


# ---------------------------------------------------------------------------
# inventory
# ---------------------------------------------------------------------------

# Rendering and filter translation live in the inventory domain modules; these aliases keep the
# historical names that the rest of this file (and the planner) use.
_vline = inventory_renderer.vline
_search_kwargs = inventory_query.search_kwargs
describe = inventory_renderer.describe


def inventory_action_payload(f: dict[str, Any]) -> dict[str, Any]:
    makes = f.get("makes") or []
    return {
        "type": "SEARCH_INVENTORY", "action_type": "SEARCH_INVENTORY",
        "make": makes[0] if len(makes) == 1 else None,
        "model": (f.get("models") or [None])[0] if len(f.get("models") or []) == 1 else None,
        "vtype": f.get("vtype") or "all",
        "fuel": f.get("fuel"), "min_price": f.get("min_price"), "max_price": f.get("max_price"),
        "body": f.get("body"), "transmission": f.get("transmission"),
    }


_STRUCTURED_MODES = {"distinct", "compare", "detail", "average", "range", "group"}


def search_inventory(f: dict[str, Any], mode: str, slots: dict[str, Any] | None = None) -> Result:
    """Deterministic inventory answer. `slots` carries the extra structured-query fields
    (field / metric / pick / compare / stock_id) for the non-search operations."""
    cat = inventory_cat()
    if mode in _STRUCTURED_MODES:
        query = inventory_query.InventoryQuery.from_slots({**(slots or {}), "filters": f}, mode)
        try:
            res = inventory_query.execute(query, cat)
        except ValueError as exc:                       # malformed structured query (e.g. from the LLM planner)
            log.warning("invalid inventory query %s: %s", query, exc)
            return Result("I couldn't work out that inventory question - could you rephrase it?", inventory_action_payload(f))
        text, ctx = inventory_renderer.render_result(res)
        return Result(text, inventory_action_payload(f), ctx=ctx)
    limit = settings.inventory_result_limit
    payload = inventory_action_payload(f)
    if mode == "unknown_brand":
        brand = (f.get("_brand") or "that brand").title()
        return Result(f"We don't stock {brand}. We carry {', '.join(cat.makes)}.")

    if mode == "makes":
        return Result(f"We carry {len(cat.makes)} makes: {', '.join(cat.makes)}.", payload)
    total, rows = cat.search(**_search_kwargs(f))
    if mode == "colors":
        colors = sorted({r["color"] for r in rows})
        if not colors:
            return Result(f"I couldn't find that vehicle in stock, so I have no colour information. We have {cat.counts()['total']} vehicles in total.", payload)
        subject = " / ".join(f.get("models") or f.get("makes") or ["these vehicles"])
        return Result(f"In stock, the {subject} comes in: {', '.join(colors)}.", payload)
    if mode == "models":
        counts: dict[str, int] = {}
        for r in rows:
            counts[r["model"]] = counts.get(r["model"], 0) + 1
        if not counts:
            return Result(f"I couldn't find any {describe(f)} in stock.", payload)
        pretty = ", ".join(f"{m} ({n})" for m, n in sorted(counts.items()))
        who = " or ".join(f.get("makes") or []) or "All"
        return Result(f"{who}: {total} vehicle{'s' if total != 1 else ''} in stock across {len(counts)} models - {pretty}.", payload)
    if mode == "superlative":
        if not rows:
            return Result(f"I couldn't find any {describe({k: v for k, v in f.items() if k != 'sort'})} in stock.", payload)
        label = {"price_asc": "cheapest", "price_desc": "most expensive", "km_asc": "lowest-mileage",
                 "year_desc": "newest", "year_asc": "oldest"}[f["sort"]]
        n = f.get("limit") or 1
        subject = describe({k: v for k, v in f.items() if k not in ("sort", "limit")}, None if n > 1 else 1)
        if n == 1:
            return Result(f"The {label} {subject} in stock:\n{_vline(rows[0])}", payload, ctx={"result_ids": [rows[0]["stock_id"]]})
        top = rows[:n]
        body = "\n".join(_vline(r) for r in top)
        return Result(f"Top {len(top)} {label} {subject} in stock:\n{body}", payload, ctx={"result_ids": [r["stock_id"] for r in top]})
    filtered = any(k for k in f if k not in ("sort",))
    if mode == "count":
        if not filtered:
            c = cat.counts()
            return Result(f"We have {c['total']} vehicles in stock: {c['new_cars']} new and {c['used_cars']} used, across {c['makes']} makes.", payload)
        return Result(f"We have {total} {describe(f, total)} in stock." if total else f"We don't have any {describe(f)} in stock right now.", payload)
    # list
    if not filtered:
        c = cat.counts()
        return Result(f"We have {c['total']} vehicles in stock ({c['new_cars']} new, {c['used_cars']} used) across {c['makes']} makes - the panel on the right lets you browse them.", payload)
    if total == 0:
        return Result(f"I couldn't find any {describe(f)} in stock. Try widening the budget or dropping a filter.", payload)
    head = f"{'Yes - w' if f.get('models') else 'W'}e have {total} {describe(f, total)} in stock."
    shown = rows[:limit]
    body = "\n".join(_vline(r) for r in shown)
    more = f"\n…and {total - limit} more - see the inventory panel." if total > limit else ""
    # --- Engaging follow-up after inventory results ---
    followup = _inventory_followup(total, f)
    return Result(f"{head}\n{body}{more}{followup}", payload, ctx={"result_ids": [r["stock_id"] for r in shown]})


def _inventory_followup(total: int, f: dict[str, Any]) -> str:
    """Generate a contextual follow-up prompt after showing inventory results."""
    if total == 1:
        return (
            "\n\n💡 Interested? I can help you:"
            "\n• **Book a test drive** for this vehicle"
            "\n• Get a **price quotation**"
            "\n• Show you **similar alternatives**"
        )
    if total <= 5:
        return (
            "\n\n💡 **What would you like to do next?**"
            "\n• **Book a test drive** — just tell me which car and your preferred date & time"
            "\n• **Compare** these vehicles side by side"
            "\n• **Narrow down** by budget, fuel type, or transmission"
        )
    return (
        "\n\n💡 **Too many options?** Try narrowing down by:"
        "\n• **Budget** — e.g. \"under 15 lakhs\""
        "\n• **Fuel type** — petrol, diesel, electric, or CNG"
        "\n• **Body type** — SUV, sedan, hatchback"
        "\n• Or just say \"**cheapest**\" or \"**newest**\""
    )


def inventory_cat():
    return inventory.catalog()