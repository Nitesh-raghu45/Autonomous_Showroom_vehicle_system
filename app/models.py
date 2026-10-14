"""Pydantic v2 request/response models for all new endpoints.

Keeping schemas here (not inline in main.py) lets route handlers stay concise
and makes it easy to generate an OpenAPI spec or share types with tests.
"""
from __future__ import annotations

import re
from datetime import date, time as dtime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from . import clock


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AppointmentType(str, Enum):
    TEST_DRIVE         = "TEST_DRIVE"
    CALLBACK           = "CALLBACK"
    ADVANCE_PAYMENT    = "ADVANCE_PAYMENT"
    DOCUMENT_SUBMISSION = "DOCUMENT_SUBMISSION"
    OTHER              = "OTHER"


class VehicleType(str, Enum):
    NEW  = "new"
    USED = "used"
    ALL  = "all"


# ---------------------------------------------------------------------------
# Customer
# ---------------------------------------------------------------------------

_MOBILE_RE = re.compile(r"^\d{10}$")


def _validate_mobile(v: str | None) -> str | None:
    if v is None:
        return v
    digits = re.sub(r"\D", "", v)[-10:]
    if len(digits) != 10:
        raise ValueError("Mobile number must be 10 digits")
    return digits


def _validate_name(v: str | None) -> str | None:
    if v is None:
        return v
    v = " ".join(v.split())
    if sum(ch.isalpha() for ch in v) < 2:
        raise ValueError("Name must contain at least 2 letters")
    if not all(ch.isalpha() or ch in " .'-" for ch in v):
        raise ValueError("Name may only contain letters, spaces, dots, apostrophes and hyphens")
    return v


class CustomerUpdateRequest(BaseModel):
    """Fields a customer is allowed to self-update."""
    name:          str | None = Field(None, min_length=2, max_length=80)
    email:         str | None = Field(None, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    mobile_no:     str | None = None      # primary (sign-in) number; uniqueness enforced in mutations
    alt_mobile_no: str | None = None
    # alt_mobile_no=None means "leave unchanged"; this flag means "remove it".
    clear_alt_mobile: bool = False

    @field_validator("name", mode="before")
    @classmethod
    def validate_name(cls, v: str | None) -> str | None:
        return _validate_name(v)

    @field_validator("alt_mobile_no", "mobile_no", mode="before")
    @classmethod
    def validate_alt_mobile(cls, v: str | None) -> str | None:
        return _validate_mobile(v)


class CustomerRegisterRequest(BaseModel):
    """Fields required to register a brand-new customer."""
    title:     str = Field("Mr", max_length=10)
    name:      str = Field(..., min_length=2, max_length=80)
    mobile_no: str
    email:     str = Field(..., pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    alt_mobile_no: str | None = None

    # Optional enquiry details at registration time
    interested_make:  str | None = None
    interested_model: str | None = None
    new_or_used:      str | None = Field(None, pattern=r"^(new|used)$")

    @field_validator("name", mode="before")
    @classmethod
    def validate_name(cls, v: str) -> str:
        return _validate_name(v)

    @field_validator("mobile_no", mode="before")
    @classmethod
    def validate_mobile(cls, v: str) -> str:
        result = _validate_mobile(v)
        if result is None:
            raise ValueError("mobile_no is required")
        return result

    @field_validator("alt_mobile_no", mode="before")
    @classmethod
    def validate_alt(cls, v: str | None) -> str | None:
        return _validate_mobile(v)


class CustomerRegisterResponse(BaseModel):
    customer_id:   str
    customer_name: str
    mobile_no:     str
    message:       str


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------

class ScheduleRequest(BaseModel):
    appointment_type: AppointmentType
    date:             date
    hour:             int = Field(..., ge=8, le=20)
    minute:           int = Field(0,  ge=0, le=59)
    notes:            str | None = Field(None, max_length=500)

    @model_validator(mode="after")
    def date_not_in_past(self) -> "ScheduleRequest":
        if self.date < clock.today():
            raise ValueError("Appointment date cannot be in the past")
        return self


class ScheduleResponse(BaseModel):
    appointment_id:   str
    customer_id:      str
    appointment_type: str
    date:             str
    time:             str
    notes:            str | None
    status:           str


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------

class InventorySearchParams(BaseModel):
    make:  str | None = None
    model: str | None = None
    type:  VehicleType = VehicleType.ALL
    fuel:  str | None = None
    min_price: int | None = None
    max_price: int | None = None


class VehicleCard(BaseModel):
    stock_id:     str
    make:         str
    model:        str
    variant:      str
    year:         int
    type:         str
    color:        str
    fuel:         str
    transmission: str
    price:        int
    mileage_km:   int


class InventoryCountResponse(BaseModel):
    total:    int
    new_cars: int
    used_cars: int
    makes:    int
