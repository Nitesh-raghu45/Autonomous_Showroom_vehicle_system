from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Deterministic intents (handled without a model)
FIELD_LOOKUP = "FIELD_LOOKUP"            # slots: fields=[...]
CONTACT_LOG = "CONTACT_LOG"              # slots: mode
APPOINTMENTS = "APPOINTMENTS"            # slots: mode
BOOK = "BOOK_APPOINTMENT"                # slots: appointment_type, when{...}, notes, problems
CANCEL = "CANCEL_APPOINTMENT"            # slots: appt_id | appointment_type | all
UPDATE = "UPDATE_INFO"                   # slots: name, email, alt_mobile_no, clear_alt, problems
REGISTER = "REGISTER_CUSTOMER"           # slots: fields + problems
INVENTORY = "SEARCH_INVENTORY"           # slots: filters, mode
SELECT_VEHICLE = "SELECT_VEHICLE"        # slots: stock_id
SQL_QUERY = "SQL_QUERY"                  # slots: question (analytical/aggregate inventory query)
NOT_IN_RECORD = "NOT_IN_RECORD"
REFUSE = "REFUSE"                        # slots: kind
CONFIRM = "CONFIRM"                      # slots: answer yes|no
GREETING = "GREETING"
HISTORY = "HISTORY"                      # "what was my last question"
CLARIFY = "CLARIFY"                      # slots: message_key, ...
NEED_CUSTOMER = "NEED_CUSTOMER"          # guest asked for something customer-specific
LLM = "LLM"                              # send to the planner


@dataclass
class RouteContext:
    customer_bound: bool = True
    customer_id: str | None = None
    customer_name: str | None = None
    has_pending: bool = False
    pending: dict[str, Any] | None = None
    own_numbers: tuple[str, ...] = ()
    previous_question: str | None = None
    last_filters: dict[str, Any] | None = None   # sticky SEARCH_INVENTORY filters from this session
    last_result_ids: list[str] | None = None      # stock_ids actually shown last turn ("these" / "among these")


@dataclass
class Route:
    intent: str
    slots: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    needs_appts: bool = False
    hints: dict[str, Any] = field(default_factory=dict)     # for the LLM path (retrieval filters etc.)

    @property
    def deterministic(self) -> bool:
        return self.intent != LLM