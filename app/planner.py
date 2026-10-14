"""LLM control plane: one constrained-JSON call -> validated plan -> shared actions.

The model decides *what* (intent + slots); `actions` does it. The `reply` field of a
plain ANSWER is streamed to the client token-by-token as the JSON is generated.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any, AsyncIterator

from functools import lru_cache

from . import actions, clock, fast_path, inventory_query
from .actions import Result
from .llm import ProviderError, get_provider
from .llm.langchain_model import ProviderChatModel
from .prompting import CHAT_PROMPT, PLAN_SCHEMA, prompt_inputs
from .router.rules import _INVENTORY_RESET
from .router.when import BUSINESS_CLOSE, BUSINESS_OPEN, When, fmt_time, hours_range

log = logging.getLogger("planner")

TEXT_INTENTS = {"ANSWER", "CLARIFY", "REFUSE"}
_UPDATE_REQUEST = re.compile(r"\b(update|change|correct|edit|modify|replace|set|add|remove|delete)\b")
_UPDATE_NEGATION = re.compile(r"\b(haven['’]?t|didn['’]?t|don['’]?t|do not|never|not)\b.{0,50}\b(update|change|correct|edit|modify|replace|set|add|remove|delete)\b")
_SELECT_REQUEST = re.compile(r"\b(go with|choose|select|pick|take)\b", re.I)
_STOCK_ID = re.compile(r"\b(?:stock\s*(?:id)?\s*[:#-]?\s*)?([A-Z]?\d{3,})\b", re.I)


class ReplyStream:
    """Incrementally pulls `intent` and the string value of `reply` out of a JSON stream."""

    def __init__(self) -> None:
        self.buf = ""
        self.intent: str | None = None
        self._pos: int | None = None       # index into buf where reply string content begins
        self._done = False
        self._emitted = 0                  # decoded characters already returned

    def feed(self, chunk: str) -> str:
        self.buf += chunk
        if self.intent is None:
            m = re.search(r'"intent"\s*:\s*"([A-Z_]+)"', self.buf)
            if m:
                self.intent = m.group(1)
        if self._pos is None:
            m = re.search(r'"reply"\s*:\s*"', self.buf)
            if not m:
                return ""
            self._pos = m.end()
        if self._done:
            return ""
        out, i, s = [], self._pos, self.buf
        while i < len(s):
            ch = s[i]
            if ch == "\\":
                if i + 1 >= len(s):
                    break
                nxt = s[i + 1]
                if nxt == "u":
                    if i + 6 > len(s):
                        break
                    try:
                        out.append(chr(int(s[i + 2:i + 6], 16)))
                    except ValueError:
                        pass
                    i += 6
                    continue
                out.append({"n": "\n", "t": "\t", "r": "\r"}.get(nxt, nxt))
                i += 2
                continue
            if ch == '"':
                self._done = True
                i += 1
                break
            out.append(ch)
            i += 1
        self._pos = i
        return "".join(out)


def _extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    a, b = text.find("{"), text.rfind("}")
    if a != -1 and b > a:
        try:
            obj = json.loads(text[a:b + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _usable_plain_reply(text: str) -> bool:
    """Accept provider prose only when it is not an empty/markdown fragment."""
    cleaned = text.strip()
    return bool(cleaned and re.search(r"[A-Za-z0-9]", cleaned) and not re.fullmatch(r"[`*_~#\s-]+", cleaned))


@dataclass
class PlanOutcome:
    result: Result | None = None
    intent: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    ttft_ms: int = 0
    llm_ms: int = 0
    error: str | None = None
    timed_out: bool = False
    streamed: bool = False       # reply tokens were already sent to the client


_AMPM_WORDS = re.compile(r"\d\s*(?:a\.?m\.?|p\.?m\.?)(?![a-z])|\bbaje\b|o'?\s?clock", re.I)


def _to_when(slots: dict[str, Any], question: str = "") -> When:
    w = When()
    # the model returns a 24-hour int; reply in AM/PM if that's how the customer wrote it
    if _AMPM_WORDS.search(question):
        w.ampm = True
    raw = slots.get("date")
    if raw:
        try:
            w.date = date.fromisoformat(str(raw)[:10])
        except ValueError:
            w.date_error = f"“{raw}” is not a valid date"
    h = slots.get("hour")
    if isinstance(h, int):
        w.hour, w.minute = h, int(slots.get("minute") or 0)
    if w.date and w.date < clock.today() and not w.date_error:
        w.date_error = f"{w.date.isoformat()} is in the past"
    if w.hour is not None and not (BUSINESS_OPEN <= w.hour <= BUSINESS_CLOSE):
        w.time_error = f"{fmt_time(w.hour, w.minute, w.ampm)} is outside our booking hours ({hours_range(w.ampm)})"
    return w


def _as_int(v: Any) -> int | None:
    """LLMs sometimes emit numbers as strings ("10", "15,00,000") - accept those, reject the rest."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, str) and re.fullmatch(r"\s*\d[\d,]*\s*", v):
        return int(v.replace(",", ""))
    return None


# The LLM sometimes fills an unnamed slot with a literal placeholder word instead of
# omitting it (e.g. make="any", fuel="any" when the person named no brand or fuel type).
# Left in, these get treated as real filter values: they match zero real vehicles and get
# stitched into the reply text itself ("any any" / "I couldn't find any any any vehicles").
_PLACEHOLDER_SLOT_VALUES = {"any", "any make", "any model", "n/a", "na", "none", "unspecified", "unknown", "all", "all makes", "all models"}


def _is_placeholder(v: Any) -> bool:
    return isinstance(v, str) and v.strip().lower() in _PLACEHOLDER_SLOT_VALUES


def _clean_slots(raw: Any) -> dict[str, Any]:
    slots = dict(raw) if isinstance(raw, dict) else {}
    for k in ("hour", "minute", "min_price", "max_price"):
        if k in slots:
            v = _as_int(slots[k])
            if v is None:
                slots.pop(k)
            else:
                slots[k] = v
    for k in list(slots):
        if slots[k] is None or (isinstance(slots[k], str) and not slots[k].strip()):
            slots.pop(k)
    return slots


async def _execute(plan: dict[str, Any], t: fast_path.Turn) -> Result:
    try:
        return await _execute_inner(plan, t)
    except Exception:  # noqa: BLE001  - a bad model plan must never become a 500
        log.exception("plan execution failed: %r", plan)
        return Result("Sorry, I couldn't complete that. Could you rephrase or give me the details again?")


async def _execute_inner(plan: dict[str, Any], t: fast_path.Turn) -> Result:
    intent = plan.get("intent")
    slots = _clean_slots(plan.get("slots"))
    reply = (plan.get("reply") or "").strip()

    if t.cid and _SELECT_REQUEST.search(t.question):
        stock = re.search(r"\bS-?\d{3,}\b", t.question, re.I)
        if stock:
            return await actions.select_vehicle(t.cid, stock.group(0).replace("-", "").upper())

    if intent in TEXT_INTENTS:
        if intent == "REFUSE":
            return Result(reply or "That's outside what I can help with here.")
        return Result(reply or "Could you rephrase that? I want to be sure I answer from your record.")

    note = f"\n\n{reply}" if reply else ""
    if intent == "REGISTER_CUSTOMER":
        if t.cid:
            from . import render
            res = Result(render.clarify("already_registered", name=(t.record or {}).get("customer", {}).get("name"), cid=t.cid))
        else:
            res = await actions.register({
                "name": slots.get("name"), "mobile_no": slots.get("mobile_no"), "email": slots.get("email"),
                "interested_make": slots.get("make"), "interested_model": slots.get("model"),
                "new_or_used": slots.get("vtype"), "problems": [],
            })
    elif intent == "SEARCH_INVENTORY":
        # The model only *describes* the query (slots); the inventory engine computes the answer.
        f, mode, extra = inventory_query.slots_to_query(slots, _is_placeholder)

        # carry forward the make/model/price/etc. from the previous turn, same as the
        # deterministic router's _h_inventory - otherwise every query that detours through
        # the LLM is a separate, context-blind search that "forgets" what was being browsed,
        # even though it shares the same renderer as the router path.
        prior = t.state.get("inv_filters") if isinstance(t.state, dict) else None
        if prior and extra.get("sticky", True) and not _INVENTORY_RESET.search(t.question.lower()):
            base = {k: v for k, v in prior.items() if k != "_brand"}
            f = {**base, **f}

        res = actions.search_inventory(f, mode, extra)
        res.ctx = {**res.ctx, "filters": f} if extra.get("sticky", True) else dict(res.ctx)
    elif not t.cid:
        from . import render
        res = Result(render.NEED_CUSTOMER_TEXT)
    elif intent == "BOOK_APPOINTMENT":
        notes = slots.get("notes") if not _is_placeholder(slots.get("notes")) else None
        res = await actions.book(t.cid, slots.get("appointment_type"), _to_when(slots, t.question), notes or fast_path.focus_vehicle_note(t),
                                 t.appts, browsed_vehicles=fast_path._get_browsed_vehicles(t.state or {}))
    elif intent == "CANCEL_APPOINTMENT":
        res = await actions.cancel_request(t.record or {}, t.appts, {
            "appt_id": slots.get("appt_id"), "appointment_type": slots.get("appointment_type"), "all": False})
    elif intent == "UPDATE_INFO":
        question = t.question.lower()
        if not _UPDATE_REQUEST.search(question) or _UPDATE_NEGATION.search(question):
            return Result("I won't change your details unless you explicitly ask me to update them.")
        res = await actions.update(t.cid, {"name": slots.get("name"), "email": slots.get("email"),
                                            "mobile_no": slots.get("mobile_no"),
                                            "alt_mobile_no": slots.get("alt_mobile_no"), "problems": [], "refused": []})
    else:
        return Result("I'm not sure how to help with that. " + "Could you rephrase?")
    if note and res.mutated:
        res.text += note
    return res


@lru_cache(maxsize=4)
def _chain_for(provider):
    """``prompt | llm`` - built once per provider, not per request."""
    return CHAT_PROMPT | ProviderChatModel(provider=provider, json_schema=PLAN_SCHEMA)


async def plan_and_execute(
    *, question: str, t: fast_path.Turn, hints: dict[str, Any] | None = None,
    turns: list[list[str]], deadline_s: float,
) -> AsyncIterator[dict[str, Any]]:
    """Async generator. Yields {"type":"token"|"outcome", ...}; the last event is the outcome."""
    outcome = PlanOutcome()
    log.info("planner.start: cid=%s question=%r deadline_s=%s hints=%s turns=%d", t.cid, question, deadline_s, hints, len(turns))
    try:
        provider = get_provider()
    except ProviderError as exc:
        outcome.error = str(exc)
        log.exception("planner.provider_error: cid=%s question=%r", t.cid, question)
        yield {"type": "outcome", "outcome": outcome}
        return

    inputs = prompt_inputs(question=question, cid=t.cid, record=t.record, appts=t.appts, turns=turns, hints=hints,
                           session=t.state if isinstance(t.state, dict) else None)
    rs = ReplyStream()
    t0 = time.perf_counter()
    try:
        stream = _chain_for(provider).astream(inputs)       # LangChain: prompt | llm, ONE call
        deadline = asyncio.get_running_loop().time() + deadline_s
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError
            try:
                ev = await asyncio.wait_for(stream.__anext__(), remaining)
            except StopAsyncIteration:
                break
            if ev.content:
                piece = rs.feed(str(ev.content))
                if piece and rs.intent in TEXT_INTENTS:
                    outcome.streamed = True
                    yield {"type": "token", "text": piece}
            um = getattr(ev, "usage_metadata", None)
            if um:
                outcome.usage = {
                    "prompt_tokens": um.get("input_tokens", 0), "completion_tokens": um.get("output_tokens", 0),
                    "cached_prompt_tokens": (um.get("input_token_details") or {}).get("cache_read", 0),
                }
                outcome.ttft_ms = int((ev.response_metadata or {}).get("ttft_ms") or 0)
    except (TimeoutError, asyncio.TimeoutError):
        outcome.timed_out = True
        log.warning("planner.timeout: cid=%s question=%r deadline_s=%s llm_ms=%s", t.cid, question, deadline_s, int((time.perf_counter() - t0) * 1000))
    except ProviderError as exc:
        outcome.error = str(exc)
        log.warning("planner.provider_error: cid=%s question=%r error=%s", t.cid, question, exc)
    except Exception as exc:  # noqa: BLE001
        log.exception("planner failure: cid=%s question=%r", t.cid, question)
        outcome.error = f"{type(exc).__name__}: {exc}"
    outcome.llm_ms = int((time.perf_counter() - t0) * 1000)

    if outcome.timed_out or outcome.error:
        yield {"type": "outcome", "outcome": outcome}
        return

    plan = _extract_json(rs.buf)
    if plan is None:
        # Non-JSON output (server ignored the schema): treat plain prose as an answer.
        text = rs.buf.strip()
        if _usable_plain_reply(text):
            outcome.intent = "ANSWER"
            outcome.result = Result(text)
        else:
            # Empty / malformed output - retry once before giving up.
            # Groq's free tier occasionally returns empty completions on short
            # conversational inputs; a single retry is cheap and usually succeeds.
            log.info("planner.retry: empty output, retrying once: cid=%s question=%r", t.cid, question)
            rs2 = ReplyStream()
            retry_ok = False
            try:
                stream2 = _chain_for(provider).astream(inputs)
                remaining2 = deadline_s - (time.perf_counter() - t0)
                if remaining2 > 2:           # only retry if we still have headroom
                    deadline2 = asyncio.get_running_loop().time() + remaining2
                    while True:
                        rem2 = deadline2 - asyncio.get_running_loop().time()
                        if rem2 <= 0:
                            break
                        try:
                            ev2 = await asyncio.wait_for(stream2.__anext__(), rem2)
                        except StopAsyncIteration:
                            break
                        if ev2.content:
                            rs2.feed(str(ev2.content))
                        um2 = getattr(ev2, "usage_metadata", None)
                        if um2:
                            outcome.usage = {
                                "prompt_tokens": um2.get("input_tokens", 0),
                                "completion_tokens": um2.get("output_tokens", 0),
                                "cached_prompt_tokens": (um2.get("input_token_details") or {}).get("cache_read", 0),
                            }
                    plan2 = _extract_json(rs2.buf)
                    if plan2 is not None:
                        outcome.intent = str(plan2.get("intent") or "ANSWER")
                        outcome.result = await _execute(plan2, t)
                        retry_ok = True
                    else:
                        text2 = rs2.buf.strip()
                        if _usable_plain_reply(text2):
                            outcome.intent = "ANSWER"
                            outcome.result = Result(text2)
                            retry_ok = True
            except Exception:  # noqa: BLE001
                log.debug("planner.retry failed", exc_info=True)
            if not retry_ok:
                outcome.error = "provider returned empty or malformed output"
        yield {"type": "outcome", "outcome": outcome}
        return
    outcome.intent = str(plan.get("intent") or "ANSWER")
    outcome.result = await _execute(plan, t)
    yield {"type": "outcome", "outcome": outcome}