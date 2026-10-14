"""REST endpoints for the Service Agent — mounted into the existing FastAPI app.

These provide direct programmatic access to service appointment management
(e.g., from an admin panel or external service center system). The chat
interface handles the same operations conversationally via the Supervisor.
"""
from __future__ import annotations

import hmac
import logging
from datetime import date
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from . import cache
from .config import settings
from .ratelimit import limiter
from .agents.service.actions import (
    _get_service_appt,
    book_service,
    cancel_service,
    list_service_appts,
    reschedule_service,
)
from .agents.service.models import ServiceSlot, ServiceType
from .agents.service.slot_store import get_available, push_update, seed_default_slots

log = logging.getLogger("service_api")

service_router = APIRouter()


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class ServiceBookRequest(BaseModel):
    service_type: ServiceType
    date: date
    hour: int = Field(..., ge=8, le=18)
    minute: int = Field(0, ge=0, le=59)
    vehicle_reg_no: Optional[str] = None
    notes: Optional[str] = None
    center: str = "default"


class ServiceRescheduleRequest(BaseModel):
    new_date: date
    new_hour: int = Field(..., ge=8, le=18)
    new_minute: int = Field(0, ge=0, le=59)
    center: Optional[str] = None


class SlotPushRequest(BaseModel):
    center: str
    date: date
    slots: list[ServiceSlot]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _resolve_or_404(number: str) -> str:
    cid = await cache.resolve_number(number)
    if cid is None:
        raise HTTPException(404, f"No customer found for '{number}'")
    return cid


def _require_webhook_secret(request: Request) -> None:
    """Slot writes come from the service-center system, never from customers. When
    SERVICE_WEBHOOK_SECRET is set, the caller must send it as X-Webhook-Secret."""
    expected = settings.service_webhook_secret
    if not expected:
        return
    given = request.headers.get("x-webhook-secret", "")
    if not hmac.compare_digest(given.encode(), expected.encode()):
        log.warning("slot webhook rejected: bad or missing secret, client=%s", request.client.host if request.client else "?")
        raise HTTPException(401, "invalid webhook secret")


# ---------------------------------------------------------------------------
# Slot routes
# ---------------------------------------------------------------------------

@service_router.get("/slots/{center}/{dt}", tags=["service-slots"])
async def get_slots(center: str, dt: date):
    """List available service slots for a center on a given date."""
    available = await get_available(center, dt)
    return {
        "center": center,
        "date": dt.isoformat(),
        "slots": [s.model_dump() for s in available],
        "count": len(available),
    }


@service_router.post("/slots/push", tags=["service-slots"])
@limiter.limit("60/minute")
async def push_slots(request: Request, body: SlotPushRequest):
    """Webhook: receive pushed slot updates from an external service system."""
    _require_webhook_secret(request)
    count = await push_update(body.center, body.date, body.slots)
    return {"status": "updated", "center": body.center, "date": body.date.isoformat(), "slots_stored": count}


@service_router.post("/slots/seed", tags=["service-slots"])
@limiter.limit("5/minute")
async def seed_slots(request: Request, center: str = "default"):
    """Seed default development slots for the next 14 days."""
    _require_webhook_secret(request)
    seeded = await seed_default_slots(center)
    return {"status": "seeded", "center": center, "days_seeded": seeded}


# ---------------------------------------------------------------------------
# Appointment routes
# ---------------------------------------------------------------------------

@service_router.post("/{customer_id}/book", tags=["service-appointments"], status_code=201)
async def book_service_appointment(customer_id: str, body: ServiceBookRequest):
    """Book a new service appointment."""
    cid = await _resolve_or_404(customer_id)
    result = await book_service(
        cid=cid,
        service_type=body.service_type.value,
        date_str=body.date.isoformat(),
        hour=body.hour,
        minute=body.minute,
        vehicle_reg_no=body.vehicle_reg_no,
        notes=body.notes,
        center=body.center,
    )
    if not result.mutated:
        raise HTTPException(400, result.text)
    return result.action.get("result") if result.action else {"message": result.text}


@service_router.patch("/{customer_id}/appointment/{appt_id}/reschedule", tags=["service-appointments"])
async def reschedule_service_appointment(customer_id: str, appt_id: str, body: ServiceRescheduleRequest):
    """Reschedule an existing service appointment."""
    cid = await _resolve_or_404(customer_id)
    result = await reschedule_service(
        cid=cid,
        appt_id=appt_id,
        new_date_str=body.new_date.isoformat(),
        new_hour=body.new_hour,
        new_minute=body.new_minute,
        center=body.center,
    )
    if not result.mutated:
        raise HTTPException(400, result.text)
    return result.action.get("result") if result.action else {"message": result.text}


@service_router.patch("/{customer_id}/appointment/{appt_id}/cancel", tags=["service-appointments"])
async def cancel_service_appointment(customer_id: str, appt_id: str):
    """Cancel an existing service appointment (immediate, no confirmation step)."""
    cid = await _resolve_or_404(customer_id)
    result = await cancel_service(cid=cid, appt_id=appt_id, confirmed=True)
    if not result.mutated:
        raise HTTPException(400, result.text)
    return result.action.get("result") if result.action else {"message": result.text}


@service_router.get("/{customer_id}/appointments", tags=["service-appointments"])
async def list_service_appointments(customer_id: str):
    """List all service appointments for a customer."""
    cid = await _resolve_or_404(customer_id)
    appts = await list_service_appts(cid)
    return {"customer_id": cid, "appointments": appts, "count": len(appts)}


@service_router.get("/{customer_id}/appointment/{appt_id}", tags=["service-appointments"])
async def get_service_appointment(customer_id: str, appt_id: str):
    """Get a single service appointment."""
    cid = await _resolve_or_404(customer_id)
    appt = await _get_service_appt(cid, appt_id)
    if appt is None:
        raise HTTPException(404, f"Service appointment {appt_id} not found")
    return appt
