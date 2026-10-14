"""Deterministic execution of routed intents: data lookup + template. No model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import actions, clock, render
from .actions import Result
from .router import intents as I
from .router.intents import Route


@dataclass
class Turn:
    question: str
    cid: str | None
    record: dict[str, Any] | None
    appts: list[dict[str, Any]]
    state: dict[str, Any]


def _get_browsed_vehicles(state: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Look up vehicles the user has browsed in this session (from inv_result_ids).

    Returns a list of vehicle dicts if the user browsed multiple cars, or None
    if they haven't browsed or only looked at one.
    """
    from . import inventory
    result_ids = state.get("inv_result_ids")
    if not result_ids or len(result_ids) < 2:
        return None
    cat = inventory.catalog()
    vehicles = []
    for sid in result_ids:
        row = next((r for r in cat.items if str(r.get("stock_id", "")) == sid), None)
        if row:
            vehicles.append(row)
    return vehicles if len(vehicles) >= 2 else None


def focus_vehicle_note(t: Turn) -> str | None:
    """Notes for a booking that names no vehicle ("book a test drive for it"): the single
    vehicle shown last turn, else the vehicle selected on the enquiry. Several browsed
    vehicles stay ambiguous - actions.book asks which one."""
    from . import inventory
    cat = inventory.catalog()
    ids = (t.state or {}).get("inv_result_ids") or []
    row = cat.get(ids[0]) if len(ids) == 1 else None
    if row is None and not ids:
        selected = ((t.record or {}).get("enquiry") or {}).get("interested_stock_id")
        row = cat.get(selected) if selected else None
    return f"{row['make']} {row['model']} ({row['stock_id']})" if row else None


async def handle(route: Route, t: Turn) -> Result:
    it, s = route.intent, route.slots
    rec = t.record or {}

    if it == I.GREETING:
        return Result(render.greeting(" ".join(t.question.strip().lower().split())))
    if it == I.HISTORY:
        prev = [x[1] for x in t.state.get("turns", []) if x[0] == "user"]
        return Result(f'Your last question was: "{prev[-1]}".' if prev else "This is your first question in this session.")
    if it == I.REFUSE:
        return Result(render.refuse(s.get("kind", "")))
    if it == I.NEED_CUSTOMER:
        return Result(render.NEED_CUSTOMER_TEXT)
    if it == I.NOT_IN_RECORD:
        return Result(render.not_in_record(topic=s.get("topic") or t.question))
    if it == I.CLARIFY:
        name = ((rec.get("customer") or {}).get("name") if t.cid else None)
        extra = {key: value for key, value in s.items() if key != "kind"}
        return Result(render.clarify(s.get("kind", ""), name=name, cid=t.cid, **extra))
    if it == I.INVENTORY:
        return actions.search_inventory(s["filters"], s["mode"], s)
    if it == I.SQL_QUERY:
        from . import sql_engine
        answer_text = await sql_engine.answer_sql(s.get("question", t.question))
        return Result(answer_text)
    if it == I.SELECT_VEHICLE:
        if not t.cid:
            return Result(render.NEED_CUSTOMER_TEXT)
        return await actions.select_vehicle(t.cid, s["stock_id"])
    if it == I.REGISTER:
        return await actions.register(s)

    # ---- everything below needs the customer's record ----
    if not t.cid or not t.record:
        return Result(render.NEED_CUSTOMER_TEXT)

    if it == I.FIELD_LOOKUP:
        return Result(render.fields_answer(s["fields"], rec, clock.today(), s))
    if it == I.CONTACT_LOG:
        return Result(render.contact_answer(rec, s))
    if it == I.APPOINTMENTS:
        return Result(render.appointments_answer(rec, t.appts))
    if it == I.BOOK:
        # Retrieve browsed vehicles from session for multi-car disambiguation
        browsed_vehicles = _get_browsed_vehicles(t.state)
        notes = s.get("notes") or focus_vehicle_note(t)
        return await actions.book(t.cid, s["appointment_type"], s["when"], notes, t.appts, browsed_vehicles=browsed_vehicles)
    if it == I.UPDATE:
        return await actions.update(t.cid, s)
    if it == I.CANCEL:
        return await actions.cancel_request(rec, t.appts, s)
    if it == I.CONFIRM:
        pending = t.state.get("pending")
        if not pending:
            return Result(render.clarify("nothing_pending"))
        if s.get("answer") == "no":
            return Result("Okay, I've left everything as it is.", clear_pending=True)
        if pending.get("type") == "UPDATE_NUMBER":
            return Result(f"Should {pending.get('number')} be your primary or your alternate number? Reply 'primary' or 'alternate'.")
        if pending.get("type") == "CANCEL_APPOINTMENT":
            return await actions.cancel_confirm(t.cid, pending)
        return Result("That request has expired - please ask again.", clear_pending=True)
    return Result(render.clarify(""))
