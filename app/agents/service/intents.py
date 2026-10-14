"""Service Agent intents and routing context.

These mirror the pattern of ``app.router.intents`` but are entirely separate
so that the Sales Agent's intent space is never polluted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ----- Intent constants -------------------------------------------------------

BOOK_SERVICE = "BOOK_SERVICE"
RESCHEDULE_SERVICE = "RESCHEDULE_SERVICE"
CANCEL_SERVICE = "CANCEL_SERVICE"
CHECK_SLOTS = "CHECK_SLOTS"
SERVICE_STATUS = "SERVICE_STATUS"
SERVICE_HISTORY = "SERVICE_HISTORY"
SERVICE_GREETING = "SERVICE_GREETING"
SERVICE_LLM = "SERVICE_LLM"               # fallback to LLM for ambiguous service queries


@dataclass
class ServiceRouteContext:
    """Context available to the service intent router."""
    customer_bound: bool = True
    customer_id: str | None = None
    customer_name: str | None = None
    has_pending: bool = False
    pending: dict[str, Any] | None = None
    active_service_appts: list[dict[str, Any]] | None = None


@dataclass
class ServiceRoute:
    """Result of the service-domain intent router."""
    intent: str
    slots: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    needs_customer: bool = False

    @property
    def deterministic(self) -> bool:
        return self.intent != SERVICE_LLM
