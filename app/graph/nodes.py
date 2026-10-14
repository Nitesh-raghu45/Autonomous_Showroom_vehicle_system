"""Graph nodes. Each node does one job and streams SSE-style events through LangGraph's
custom stream writer, so the HTTP layer sees exactly the events it always did:

    meta -> token* -> [replace] -> [action] -> done          (or:  error)

    load_state    verify session / customer once, ONE Redis pipeline (session+history, record, appts)
    route         deterministic router (pure function, microseconds)
    rule_handler  data lookup / validated action / template      -> NO LLM
    llm_handler   LangChain `prompt | llm`, ONE call              -> only when the router defers
    save_history  side effects, append the turn, write back to Redis (fire-and-forget), `done`
"""
from __future__ import annotations

import logging
import time
from typing import Any

from langgraph.config import get_stream_writer

from .. import cache, fast_path, guard, observability, planner
from ..actions import Result
from ..config import settings
from ..l1 import TTLCache
from ..prompting import answer_cache_key
from ..router import RouteContext, route as run_router
from ..router import intents as I
from ..router import normalize
from .state import ChatState

log = logging.getLogger("graph")

_answers = TTLCache(2000, 600)          # in-process answer cache, LLM path only
REPLAY_CHUNK = 24
GUEST = {"", "guest", "none", "null"}

FALLBACK_TIMEOUT = ("I’m still ready to answer from your customer record and recent conversation. Please try the question again in a simpler form and I’ll use your details to answer it.")
FALLBACK_ERROR = ("I couldn't complete that response just now. Your customer record, appointments, and stock details are still available; please try again shortly.")


def _ev(event: str, **data: Any) -> dict[str, Any]:
    return {"event": event, "data": data}


def _ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


def _turn(state: ChatState) -> fast_path.Turn:
    return fast_path.Turn(
        question=state["question"], cid=state["cid"], record=state["record"],
        appts=state["appts"], state=state["session"],
    )


# ---------------------------------------------------------------------------
# 1. load_state - verify once, one Redis round trip
# ---------------------------------------------------------------------------

async def load_state(state: ChatState) -> dict[str, Any]:
    write = get_stream_writer()
    t0 = time.perf_counter()
    session_id, question = state["session_id"], state["question"]
    timings: dict[str, int] = {}
    log.info("graph.load_state: session_id=%s customer_number=%s question=%r", session_id, state.get("customer_number"), question)

    def fail(message: str) -> dict[str, Any]:
        write(_ev("error", message=message))
        return {"error": message, "t0": t0, "timings": timings}

    # A customer is verified at session start. Once bound, the session id IS the identity for every
    # following turn; the client does not resend the mobile/customer number. If a number is supplied
    # again it must resolve to the same customer (prevents cross-customer context leakage).
    number = (state.get("customer_number") or "").strip()
    cid: str | None = None
    bound_cid = await cache.session_customer(session_id) if session_id else None
    if bound_cid:
        cid = bound_cid
        if number.lower() not in GUEST:
            supplied_cid = await cache.resolve_number(number)
            if supplied_cid != cid:
                return fail("This session is already verified for a different customer. Please start a new session.")
        log.info("verified session: session_id=%s cid=%s", session_id, cid)
    elif number.lower() not in GUEST:
        cid = await cache.resolve_number(number)
        if cid is None:
            log.warning("customer lookup failed: session_id=%s customer_number=%s", session_id, number)
            return fail(f"No customer found for '{number}'")
        await cache.bind_session(session_id, cid)
        log.info("customer verified and session bound: session_id=%s cid=%s", session_id, cid)
    else:
        log.info("guest session: session_id=%s", session_id)

    b = await cache.fetch_bundle(cid, session_id, want_appts=bool(cid))
    timings["data_ms"] = _ms(t0)
    session = b["state"] or cache.new_state(cid)
    log.info("session bundle loaded: session_id=%s cid=%s turns=%d appts=%d record_src=%s redis=%s",
             session_id, cid, len(session.get("turns", [])), len(b.get("appts") or []), b.get("record_src"), b.get("redis"))
    if cid and session.get("cid") and session["cid"] != cid:
        return fail("This session belongs to a different customer. Please start a new session.")
    session["cid"] = cid or session.get("cid")
    if cid and b["record"] is None:
        return fail(f"No customer found for '{number}'")

    pending = session.get("pending")
    if pending:
        pending["turns_left"] = int(pending.get("turns_left", 1))
    return {
        "t0": t0, "timings": timings, "cid": cid, "record": b["record"], "appts": b["appts"],
        "session": session, "turns": session.get("turns", []), "pending": pending,
        "record_src": b["record_src"], "redis_up": b["redis"], "error": None,
    }


def after_load(state: ChatState) -> str:
    return "end" if state.get("error") else "route"


# ---------------------------------------------------------------------------
# 2. route - deterministic, no model
# ---------------------------------------------------------------------------

async def route_question(state: ChatState) -> dict[str, Any]:
    write = get_stream_writer()
    t1 = time.perf_counter()
    record, cid, pending, turns = state["record"], state["cid"], state["pending"], state["turns"]
    customer = (record or {}).get("customer") or {}
    previous_question = next((text for role, text in reversed(turns) if role == "user"), None)
    ctx = RouteContext(
        customer_bound=bool(cid), customer_id=cid, customer_name=customer.get("name"),
        has_pending=bool(pending), pending=pending,
        own_numbers=tuple(n for n in (customer.get("mobile_no"), customer.get("alt_mobile_no")) if n),
        previous_question=previous_question, last_filters=state["session"].get("inv_filters"),
        last_result_ids=state["session"].get("inv_result_ids"),
    )
    rt = run_router(state["question"], ctx) if settings.router_enabled else I.Route(I.LLM, {}, "router disabled", needs_appts=True)
    route_ms = int((time.perf_counter() - t1) * 1000)
    log.info("route decision: session_id=%s cid=%s intent=%s reason=%s route_ms=%s", state["session_id"], cid, rt.intent, rt.reason, route_ms)

    is_llm = rt.intent == I.LLM
    write(_ev(
        "meta", customer_id=cid, customer_name=customer.get("name"), route="llm" if is_llm else "router",
        intent=rt.intent, reason=rt.reason, profile_cache=state["record_src"], redis=state["redis_up"],
        intent_classifier=None,
        provider=settings.llm_provider if is_llm else None,
        model=settings.model_name if is_llm else None, history_turns=len(turns) // 2,
        summary_active=False,
    ))
    return {"decision": rt, "timings": {**state["timings"], "route_ms": route_ms}}


def route_decision(state: ChatState) -> str:
    return "llm" if state["decision"].intent == I.LLM else "rule"


# ---------------------------------------------------------------------------
# 3a. rule_handler - NO LLM
# ---------------------------------------------------------------------------

async def rule_handler(state: ChatState) -> dict[str, Any]:
    write = get_stream_writer()
    rt, sid = state["decision"], state["session_id"]
    t2 = time.perf_counter()
    log.info("deterministic path: session_id=%s intent=%s slots=%s", sid, rt.intent, rt.slots)
    try:
        result = await fast_path.handle(rt, _turn(state))
    except Exception:  # noqa: BLE001  - never surface a stack trace to the user
        log.exception("deterministic handler failed for intent %s session_id=%s", rt.intent, sid)
        observability.bump("handler_error_total")
        result = Result("Sorry, something went wrong on my side. Please try again.")
    exec_ms = _ms(t2)
    log.info("deterministic path complete: session_id=%s intent=%s text=%r exec_ms=%s", sid, rt.intent, result.text, exec_ms)
    write(_ev("token", text=result.text))
    return {"result": result, "source": "router", "timings": {**state["timings"], "exec_ms": exec_ms}}


# ---------------------------------------------------------------------------
# 3b. llm_handler - customer JSON + bounded history + question -> ONE LLM call
# ---------------------------------------------------------------------------

async def llm_handler(state: ChatState) -> dict[str, Any]:
    write = get_stream_writer()
    rt, sid, cid = state["decision"], state["session_id"], state["cid"]
    question, record, appts, turns = state["question"], state["record"], state["appts"], state["turns"]
    timings = dict(state["timings"])
    previous_question = next((text for role, text in reversed(turns) if role == "user"), None)
    previous_answer = next((text for role, text in reversed(turns) if role == "assistant"), None)

    source = "llm"
    result: Result | None = None
    usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "cached_prompt_tokens": 0}
    ttft_ms = llm_ms = 0
    planner_intent: str | None = None

    ck = None
    if settings.llm_answer_cache and cid and record:
        ck = answer_cache_key(cid, record, appts, question, turns)
        hit = _answers.get(ck)
        if hit is not None:
            source = "answer_cache"
            result = Result(hit)
            for i in range(0, len(hit), REPLAY_CHUNK):
                write(_ev("token", text=hit[i:i + REPLAY_CHUNK]))

    if result is None:
        streamed = ""
        outcome = None
        log.info("LLM path start: session_id=%s cid=%s deadline_ms=%s question=%r", sid, cid, settings.llm_deadline_ms, question)
        async for ev in planner.plan_and_execute(
            question=normalize.correct(question), t=_turn(state), hints=rt.hints, turns=turns,
            deadline_s=settings.llm_deadline_ms / 1000,
        ):
            if ev["type"] == "token":
                streamed += ev["text"]
                write(_ev("token", text=ev["text"]))
            else:
                outcome = ev["outcome"]
        usage, ttft_ms, llm_ms = outcome.usage or usage, outcome.ttft_ms, outcome.llm_ms
        planner_intent = outcome.intent
        timings["llm_ms"] = llm_ms
        log.info("LLM outcome: session_id=%s intent=%s timed_out=%s error=%s llm_ms=%s usage=%s", sid, planner_intent, bool(outcome.timed_out), outcome.error, llm_ms, usage)
        if outcome.timed_out:
            source, result = "llm_timeout", Result(FALLBACK_TIMEOUT)
            observability.bump("llm_timeout_total")
            log.warning("LLM timed out: session_id=%s question=%r deadline_ms=%s", sid, question, settings.llm_deadline_ms)
        elif outcome.error or outcome.result is None:
            source, result = "llm_error", Result(FALLBACK_ERROR)
            observability.bump("llm_error_total")
            log.warning("llm error: session_id=%s error=%s question=%r", sid, outcome.error, question)
        else:
            result = outcome.result
            if (
                outcome.intent in planner.TEXT_INTENTS
                and guard.is_degenerate_repeat(question, previous_question, previous_answer, result.text)
            ):
                log.warning("planner produced a context-blind repeat: session_id=%s question=%r reused_answer=%r", sid, question, result.text)
                observability.bump("llm_degenerate_repeat_total")
                source, result = "llm_degenerate_repeat", Result(guard.GENERIC_CLARIFY)
            elif not outcome.result.mutated and outcome.intent in planner.TEXT_INTENTS and ck:
                _answers.set(ck, result.text)
        if streamed and streamed != result.text:
            write(_ev("replace", text=result.text))
        elif not streamed:
            write(_ev("token", text=result.text))

    return {
        "result": result, "source": source, "usage": usage, "ttft_ms": ttft_ms, "llm_ms": llm_ms,
        "planner_intent": planner_intent, "timings": timings,
    }


# ---------------------------------------------------------------------------
# 4. save_history - side effects, full history persisted to Redis, `done`
# ---------------------------------------------------------------------------

async def save_history(state: ChatState) -> dict[str, Any]:
    write = get_stream_writer()
    rt, result, session, sid, cid = state["decision"], state["result"], state["session"], state["session_id"], state["cid"]
    question, turns, pending = state["question"], state["turns"], state["pending"]
    planner_intent = state.get("planner_intent")

    if result.action:
        log.info("action emitted: session_id=%s action=%s", sid, result.action)
        write(_ev("action", **result.action))

    if result.clear_pending:
        session["pending"] = None
    elif result.pending:
        session["pending"] = result.pending
    elif pending:
        pending["turns_left"] -= 1
        session["pending"] = pending if pending["turns_left"] > 0 else None

    # Keep the browsing context (make/model/price/...) sticky across turns so a bare refinement like
    # "50 lacs above and new" is understood - but drop it once the person moves on to a transaction.
    # sort/limit are one-off directives ("top 3", "cheapest"), so they are deliberately not remembered.
    # This must fire for BOTH the deterministic router (rt.intent) and the LLM planner (planner_intent):
    # when a turn detours through the LLM, rt.intent stays "LLM" but the planner may still have run
    # SEARCH_INVENTORY (or a transaction) internally.
    resolved_intent = rt.intent if rt.intent != I.LLM else planner_intent
    if resolved_intent == I.INVENTORY and rt.intent == I.INVENTORY and rt.slots.get("sticky") is False:
        # whole-catalogue analytics / single-vehicle lookups must not overwrite the browsing filters -
        # but a single vehicle the person just asked about IS now what "it" / "book a test drive" means
        if rt.slots.get("mode") == "detail" and result.ctx.get("result_ids"):
            session["inv_result_ids"] = result.ctx["result_ids"]
    elif resolved_intent == I.INVENTORY:
        filters = rt.slots.get("filters") if rt.intent == I.INVENTORY else result.ctx.get("filters")
        if isinstance(filters, dict):
            session["inv_filters"] = {k: v for k, v in filters.items() if k not in ("_brand", "stock_ids", "sort", "limit")}
            # only overwrite the shown result set when this turn surfaced concrete vehicles - a
            # meta-query like "what colours" shouldn't wipe out what "these" still refers to.
            if "result_ids" in result.ctx:
                session["inv_result_ids"] = result.ctx["result_ids"]
    elif resolved_intent in (I.BOOK, I.UPDATE, I.REGISTER, I.CANCEL, I.SELECT_VEHICLE):
        session["inv_filters"] = None
        session["inv_result_ids"] = None

    # The COMPLETE history is appended here and persisted; history.py only shapes what the LLM sees.
    # A failed/timed-out LLM turn is infrastructure noise, not conversation: storing the canned fallback as an
    # "assistant" message pollutes the context the model sees on the user's retry.
    if state["source"] not in ("llm_error", "llm_timeout"):
        session["turns"] = turns + [["user", question], ["assistant", result.text]]
    log.info("persisting session state: session_id=%s cid=%s pending=%s inv_filters=%s", sid, cid, bool(session.get("pending")), bool(session.get("inv_filters")))
    cache.save_state_bg(sid, session)

    total = _ms(state["t0"])
    timings = {**state["timings"], "total_ms": total}
    source = state["source"]
    observability.record(source, total)
    observability.bump(f"intent_total|{rt.intent}")
    is_llm = rt.intent == I.LLM
    write(_ev(
        "done", source=source, route="router" if source == "router" else "llm", intent=rt.intent,
        intent_classifier=None,
        latency_ms=total, ttft_ms=state.get("ttft_ms", 0), llm_ms=state.get("llm_ms", 0), timings=timings,
        usage=state.get("usage") or {"prompt_tokens": 0, "completion_tokens": 0, "cached_prompt_tokens": 0},
        provider=settings.llm_provider if is_llm else None,
        model=settings.model_name if is_llm else None,
        summarised=False, sla_met=total <= 500,
    ))
    return {"timings": timings}