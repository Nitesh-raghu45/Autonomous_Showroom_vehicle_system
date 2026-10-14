"""Write operations. Each one is a single atomic Redis step (Lua) or an optimistic CAS
read-modify-write - no multi-command sequences that can interleave or half-apply.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from . import cache, clock, persistence
from .config import settings
from .models import CustomerRegisterRequest, CustomerUpdateRequest, ScheduleRequest

log = logging.getLogger("mutations")


class Conflict(ValueError):
    """Business-rule conflict (duplicate mobile etc.) - message is safe to show to the user."""


async def update_customer(customer_id: str, req: CustomerUpdateRequest) -> dict[str, Any]:
    changed: list[str] = []
    old: dict[str, str | None] = {}
    claimed: list[str] = []

    current = await cache.get_customer_record(customer_id)
    if current is None:
        raise KeyError(f"Customer {customer_id} not found")
    cur = current.get("customer") or {}

    # Uniqueness: a number can belong to exactly one account, primary or alternate.
    for label, number in (("primary", req.mobile_no), ("alternate", req.alt_mobile_no)):
        if not number:
            continue
        if number == cur.get("mobile_no") and label == "alternate":
            raise Conflict("That's already your primary mobile number.")
        if number == cur.get("alt_mobile_no") and label == "primary":
            raise Conflict("That's already your alternate number - remove it first if you want it as your primary.")
        if number in (cur.get("mobile_no"), cur.get("alt_mobile_no")):
            continue                                     # unchanged
        try:
            ok = await cache.claim_mobile(number, customer_id)
        except cache.StoreUnavailable:
            for n in claimed:
                await cache.release_mobile(n, customer_id)
            raise
        if not ok:
            for n in claimed:
                await cache.release_mobile(n, customer_id)
            raise Conflict("That number is already registered to another account, so it can't be used.")
        claimed.append(number)

    def apply(record: dict[str, Any]) -> dict[str, Any]:
        changed.clear()
        old.clear()
        customer = dict(record.get("customer") or {})
        if req.name is not None and req.name != customer.get("name"):
            customer["name"] = req.name
            changed.append("name")
        if req.email is not None and req.email != customer.get("email"):
            customer["email"] = req.email
            changed.append("email")
        if req.mobile_no and req.mobile_no != customer.get("mobile_no"):
            old["mobile_no"] = customer.get("mobile_no")
            customer["mobile_no"] = req.mobile_no
            changed.append("mobile_no")
        if req.clear_alt_mobile and customer.get("alt_mobile_no"):
            old["alt_mobile_no"] = customer.get("alt_mobile_no")
            customer["alt_mobile_no"] = None
            changed.append("alt_mobile_no")
        elif req.alt_mobile_no is not None and req.alt_mobile_no != customer.get("alt_mobile_no"):
            old["alt_mobile_no"] = customer.get("alt_mobile_no")
            customer["alt_mobile_no"] = req.alt_mobile_no
            changed.append("alt_mobile_no")
        return {**record, "customer": customer}

    try:
        updated = await cache.update_customer_record(customer_id, apply)
    except Exception:
        for n in claimed:
            await cache.release_mobile(n, customer_id)
        raise
    for key, prev in old.items():                        # free the numbers this customer gave up
        if prev:
            await cache.release_mobile(prev, customer_id)
    persistence.enqueue("customer", customer_id, updated)
    log.info("customer %s updated fields: %s", customer_id, changed)
    return {"customer_id": customer_id, "updated_fields": list(changed), "record": updated}


async def select_vehicle(customer_id: str, row: dict[str, Any]) -> dict[str, Any]:
    def apply(record: dict[str, Any]) -> dict[str, Any]:
        enquiry = dict(record.get("enquiry") or {})
        enquiry.update({
            "interested_make": row.get("make"),
            "interested_model": row.get("model"),
            "interested_stock_id": row.get("stock_id"),
        })
        return {**record, "enquiry": enquiry}

    updated = await cache.update_customer_record(customer_id, apply)
    persistence.enqueue("customer", customer_id, updated)
    return {"customer_id": customer_id, "updated_fields": ["interested_vehicle"], "record": updated}


async def register_customer(req: CustomerRegisterRequest) -> dict[str, Any]:
    enquiry_id = f"E-{uuid.uuid4().hex[:8].upper()}"
    template: dict[str, Any] = {
        "customer": {
            "customer_id": "__CID__", "title": req.title, "name": req.name,
            "mobile_no": req.mobile_no, "alt_mobile_no": req.alt_mobile_no, "email": req.email,
        },
        "enquiry": {
            "enquiry_id": enquiry_id, "salesperson": None, "created_at": clock.today().isoformat(),
            "status": "open", "new_or_used": req.new_or_used,
            "interested_make": req.interested_make, "interested_model": req.interested_model,
        },
        "contact_log": [], "quotation": None, "scheduled_contact": None,
    }
    status, cid = await cache.register_customer_atomic(template, req.mobile_no, req.alt_mobile_no)
    if status != "created":
        raise Conflict("That mobile number is already registered. Please start a session with it to access your account.")
    template["customer"]["customer_id"] = cid
    cache.remember_customer(cid, template)
    persistence.enqueue("customer", cid, template)
    log.info("new customer registered: id=%s", cid)
    return {
        "customer_id": cid, "customer_name": req.name, "mobile_no": req.mobile_no,
        "enquiry_id": enquiry_id, "message": f"Customer registered successfully. Your ID is {cid}.",
    }


async def book_appointment(customer_id: str, req: ScheduleRequest) -> dict[str, Any]:
    appt_id = uuid.uuid4().hex[:12].upper()
    appt: dict[str, Any] = {
        "appointment_id": appt_id, "customer_id": customer_id,
        "appointment_type": req.appointment_type.value, "date": req.date.isoformat(),
        "time": f"{req.hour:02d}:{req.minute:02d}", "notes": req.notes, "status": "confirmed",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
    }
    if not await cache.claim_slot(customer_id, appt["date"], appt["time"], appt_id):
        raise Conflict("You already have an appointment at that date and time.")
    atype = appt["appointment_type"]
    try:
        if not await cache.claim_capacity(atype, appt["date"], appt["time"], appt_id, settings.slot_capacity(atype)):
            raise Conflict(f"Sorry, {appt['date']} at {appt['time']} is fully booked. Please pick another time.")
    except BaseException:
        await cache.release_slot(customer_id, appt["date"], appt["time"], appt_id)
        raise
    try:
        await cache.save_appointment(customer_id, appt)
    except Exception:
        await cache.release_slot(customer_id, appt["date"], appt["time"], appt_id)
        await cache.release_capacity(atype, appt["date"], appt["time"], appt_id)
        raise
    persistence.enqueue("appointment", f"{customer_id}:{appt_id}", appt)
    return appt


async def cancel_appointment(customer_id: str, appt_id: str) -> dict[str, Any]:
    appt = await cache.get_appointment(customer_id, appt_id)
    if appt is None:
        raise KeyError(f"Appointment {appt_id} not found for customer {customer_id}")
    appt["status"] = "cancelled"
    await cache.save_appointment(customer_id, appt)
    await cache.release_slot(customer_id, appt.get("date", ""), appt.get("time", ""), appt_id)
    await cache.release_capacity(appt.get("appointment_type", ""), appt.get("date", ""), appt.get("time", ""), appt_id)
    persistence.enqueue("appointment", f"{customer_id}:{appt_id}", appt)
    return appt
