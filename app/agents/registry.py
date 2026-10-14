"""Agent registry — enumerates all agent domains and provides lookup."""
from __future__ import annotations

from enum import Enum


class AgentDomain(str, Enum):
    """The three agent domains the supervisor can route to."""
    SALES = "sales"
    SERVICE = "service"
    MANAGEMENT = "management"
