"""LangGraph state for one chat turn.

The verified customer record, appointments and the Redis session (which holds the complete
history) are loaded ONCE by ``load_state`` and then travel through the graph, so no node ever
re-reads Redis or the JSON store.

NOTE: LangGraph forbids a state key that has the same name as a node, so the routing decision is
stored under ``decision`` (the node is called ``route``).
"""
from __future__ import annotations

from typing import Any, TypedDict

from ..actions import Result
from ..router.intents import Route


class ChatState(TypedDict, total=False):
    # ---- input -----------------------------------------------------------------
    session_id: str
    customer_number: str
    question: str

    # ---- load_state ------------------------------------------------------------
    t0: float                         # perf_counter at request start
    timings: dict[str, int]
    cid: str | None                   # verified customer id (None = guest)
    record: dict[str, Any] | None     # the customer's complete JSON record
    appts: list[dict[str, Any]]
    session: dict[str, Any]           # compact Redis session: turns (full history), pending, inv_filters ...
    turns: list[list[str]]            # == session["turns"] at load time
    pending: dict[str, Any] | None
    record_src: str
    redis_up: bool
    error: str | None                 # set => the graph ends after load_state (an `error` event was emitted)

    # ---- route -----------------------------------------------------------------
    decision: Route

    # ---- handlers --------------------------------------------------------------
    result: Result
    source: str                       # router | answer_cache | llm | llm_timeout | llm_error | llm_degenerate_repeat
    usage: dict[str, int]
    ttft_ms: int
    llm_ms: int
    planner_intent: str | None        # the LLM planner's own intent when the router said LLM
