"""Pydantic models for the Service Agent domain.

Separate from ``app.models`` so the sales-agent schemas are never touched.
"""
from __future__ import annotations

from datetime import date, time as dtime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator

from ... import clock


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ServiceType(str, Enum):
    """Types of vehicle service appointments."""
    FREE_SERVICE = "FREE_SERVICE"
    PAID_SERVICE = "PAID_SERVICE"
    PERIODIC_MAINTENANCE = "PERIODIC_MAINTENANCE"
    BODY_REPAIR = "BODY_REPAIR"
    ACCIDENTAL_REPAIR = "ACCIDENTAL_REPAIR"
    WARRANTY_CLAIM = "WARRANTY_CLAIM"
    GENERAL = "GENERAL"


class ServiceAppointmentStatus(str, Enum):
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    RESCHEDULED = "rescheduled"
    NO_SHOW = "no_show"


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ServiceSlot(BaseModel):
    """A single available service slot at a center."""
    center: str
    date: date
    time: str                      # HH:MM
    capacity: int = 1              # how many vehicles can be taken in this slot
    available: int = 1             # remaining capacity
    service_types: list[str] = Field(default_factory=list)  # empty = accepts all types

    @property
    def is_available(self) -> bool:
        return self.available > 0


class ServiceAppointmentRequest(BaseModel):
    """Request to book a service appointment."""
    service_type: ServiceType
    date: date
    hour: int = Field(..., ge=8, le=18)
    minute: int = Field(0, ge=0, le=59)
    vehicle_reg_no: Optional[str] = Field(None, max_length=20)
    notes: Optional[str] = Field(None, max_length=500)
    center: str = "default"

    @model_validator(mode="after")
    def date_not_in_past(self) -> "ServiceAppointmentRequest":
        if self.date < clock.today():
            raise ValueError("Service appointment date cannot be in the past")
        return self


class ServiceRescheduleRequest(BaseModel):
    """Request to reschedule an existing service appointment."""
    new_date: date
    new_hour: int = Field(..., ge=8, le=18)
    new_minute: int = Field(0, ge=0, le=59)
    center: Optional[str] = None     # None = keep same center

    @model_validator(mode="after")
    def date_not_in_past(self) -> "ServiceRescheduleRequest":
        if self.new_date < clock.today():
            raise ValueError("Rescheduled date cannot be in the past")
        return self


# ---------------------------------------------------------------------------
# Type labels (for human-readable output)
# ---------------------------------------------------------------------------

SERVICE_TYPE_LABEL: dict[str, str] = {
    "FREE_SERVICE": "Free Service",
    "PAID_SERVICE": "Paid Service",
    "PERIODIC_MAINTENANCE": "Periodic Maintenance",
    "BODY_REPAIR": "Body Repair",
    "ACCIDENTAL_REPAIR": "Accidental Repair",
    "WARRANTY_CLAIM": "Warranty Claim",
    "GENERAL": "General Service",
}
