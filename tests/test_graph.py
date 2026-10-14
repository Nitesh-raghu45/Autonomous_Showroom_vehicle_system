"""LangGraph workflow: topology, rule path (no LLM), LLM path (ONE LangChain call), history, verification."""
import dataclasses
import json

import httpx
import pytest
import pytest_asyncio

from app import cache, planner, service
from app.config import settings
from app.graph import get_graph

pytestmark = pytest.mark.asyncio(loop_scope="module")

LLM_QUESTION = "Compare a new Kia Seltos and a used BMW X1 — which is cheaper?"   # no rule matches -> LLM


@pytest_asyncio.fixture(scope="module", autouse=True)
async def app_lifespan():
    from app.llm import http as llm_http
    from app.main import app
    from scripts import mock_llm_server

    llm_http._client = httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_llm_server.app), base_url="http://mock-llm")
    async with app.router.lifespan_context(app):
        yield


class RecordingProvider:
    """Stands in for the LLM server. Records every prompt it receives."""
    name, model = "recording", "recording-1"

    def __init__(self, reply="The Kia Seltos is cheaper than the BMW X1 in the stock rows."):
        self.calls: list[list[dict]] = []
        self.reply = reply

    async def stream(self, messages, *, schema=None, max_tokens=None):
        self.calls.append(messages)
        text = json.dumps({"intent": "ANSWER", "reply": self.reply, "slots": {}})
        for i in range(0, len(text), 10):
            yield {"type": "token", "text": text[i:i + 10]}
        yield {"type": "usage", "usage": {"prompt_tokens": 123, "completion_tokens": 9, "cached_prompt_tokens": 50}, "ttft_ms": 17}

    async def warmup(self, messages):
        return None


@pytest.fixture
def llm(monkeypatch):
    prov = RecordingProvider()
    monkeypatch.setattr(planner, "get_provider", lambda: prov)
    return prov


def patch_settings(monkeypatch, module, **overrides):
    """`settings` is a frozen dataclass: swap in a modified copy for one module."""
    monkeypatch.setattr(module, "settings", dataclasses.replace(module.settings, **overrides))


async def run(sid, number, question):
    return [ev async for ev in service.answer(sid, number, question)]


def names(events):
    return [e["event"] for e in events]


def text(events):
    out = ""
    for e in events:
        if e["event"] == "token":
            out += e["data"]["text"]
        elif e["event"] == "replace":
            out = e["data"]["text"]
    return out


# ---- topology -------------------------------------------------------------------------

async def test_graph_topology_matches_the_architecture():
    g = get_graph().get_graph()
    assert set(g.nodes) == {"__start__", "load_state", "route", "rule_handler", "llm_handler", "save_history", "__end__"}
    edges = {(e.source, e.target) for e in g.edges}
    assert {("__start__", "load_state"), ("load_state", "route"), ("load_state", "__end__"),
            ("route", "rule_handler"), ("route", "llm_handler"),
            ("rule_handler", "save_history"), ("llm_handler", "save_history"), ("save_history", "__end__")} <= edges


# ---- rule path: no LLM -------------------------------------------------------------------

async def test_rule_path_never_calls_the_llm(monkeypatch):
    async def boom(**kwargs):
        raise AssertionError("LLM path must not run for a rule-answerable question")
        yield  # pragma: no cover

    monkeypatch.setattr(planner, "plan_and_execute", boom)
    evs = await run("g-rule", "10001", "What's my customer ID?")
    assert names(evs) == ["meta", "token", "done"]
    assert "10001" in text(evs)
    done = evs[-1]["data"]
    assert done["source"] == "router" and done["llm_ms"] == 0 and done["usage"]["completion_tokens"] == 0
    assert evs[0]["data"]["route"] == "router" and evs[0]["data"]["provider"] is None


async def test_action_event_is_emitted_before_done():
    evs = await run("g-act", "10004", "Can you book a test drive for me on 25th September at 10am?")
    assert names(evs) == ["meta", "token", "action", "done"]
    assert evs[2]["data"]["type"] == "BOOK_APPOINTMENT"


# ---- LLM path: ONE call, customer JSON + history + question ---------------------------------

async def test_llm_path_makes_exactly_one_call_with_json_history_and_question(llm):
    sid = "g-llm"
    await run(sid, "10001", "What's my customer ID?")                 # rule-answered, still enters history
    evs = await run(sid, "", LLM_QUESTION)

    assert len(llm.calls) == 1
    msgs = llm.calls[0]
    assert msgs[0]["role"] == "system" and "Customer Assist planner" in msgs[0]["content"]
    ctx = next(m["content"] for m in msgs if m["role"] == "system" and "CUSTOMER_JSON" in m["content"])
    rec = await cache.get_customer_record("10001")
    assert rec["customer"]["name"] in ctx and "authoritative DATA" in ctx
    convo = [m for m in msgs if m["role"] in ("user", "assistant")]
    assert [m["role"] for m in convo] == ["user", "assistant", "user"]
    assert convo[0]["content"] == "What's my customer ID?" and "10001" in convo[1]["content"]
    assert convo[-1]["content"] == LLM_QUESTION

    assert names(evs)[0] == "meta" and names(evs)[-1] == "done"
    assert evs[0]["data"]["route"] == "llm" and evs[0]["data"]["provider"] == settings.llm_provider
    assert "Seltos" in text(evs)
    done = evs[-1]["data"]
    assert done["source"] == "llm" and done["ttft_ms"] == 17
    assert done["usage"] == {"prompt_tokens": 123, "completion_tokens": 9, "cached_prompt_tokens": 50}


@pytest.mark.parametrize("question", [
    "can you tell me about my quotaion",
    "findout my nmae",
])
async def test_misspelled_question_reaches_grounded_model_unchanged(llm, question):
    llm.reply = "I checked the customer record for you."
    evs = await run(f"g-typo-{question[-4:]}", "10001", question)

    assert evs[0]["data"]["route"] == "llm"
    assert evs[-1]["data"]["source"] == "llm"
    assert len(llm.calls) == 1
    messages = llm.calls[0]
    assert messages[-1] == {"role": "user", "content": question}
    context = next(m["content"] for m in messages if m["role"] == "system" and "CUSTOMER_JSON" in m["content"])
    assert "Amit Verma" in context
    rules = next(m["content"] for m in messages if m["role"] == "system" and "Customer Assist planner" in m["content"])
    assert "spelling mistakes, typos" in rules


async def test_complete_history_is_stored_but_the_prompt_is_bounded(llm, monkeypatch):
    from app import history
    patch_settings(monkeypatch, history, llm_history_recent_turns=2, llm_history_compact_chars=20, llm_history_max_tokens=0)
    sid = "g-long"
    for q in ["What's my customer ID?", "what is my email?", "what is my mobile number?", "who is my salesperson?",
              "what is my name?", "what is my enquiry status?"]:
        await run(sid, "10001", q)
    await run(sid, "", LLM_QUESTION)

    stored = (await cache.load_state(sid))["turns"]
    assert len(stored) == 14                                          # 7 exchanges, nothing lost
    before = stored[:12]                                              # what existed when the LLM turn started
    convo = [m for m in llm.calls[0] if m["role"] in ("user", "assistant")]
    assert len(convo) == 13                                           # all 12 earlier messages + the question
    assert [m["content"] for m in convo[-5:-1]] == [t[1] for t in before[-4:]]      # last 2 exchanges verbatim
    older = convo[:-5]
    assert all(len(m["content"]) <= 20 for m in older)                # older ones shortened...
    assert any(m["content"].endswith("…") for m in older)             # ...and at least one really was


async def test_history_tail_is_dropped_only_by_the_token_cap(llm, monkeypatch):
    from app import history
    patch_settings(monkeypatch, history, llm_history_recent_turns=100, llm_history_max_tokens=150)
    sid = "g-cap"
    for _ in range(8):
        await run(sid, "10001", "what is my email?")
    await run(sid, "", LLM_QUESTION)
    stored = (await cache.load_state(sid))["turns"]
    assert len(stored) == 18                                          # lossless in Redis
    sent = llm.calls[0]
    assert any("omitted" in m["content"] for m in sent if m["role"] == "system")
    assert sent[-1] == {"role": "user", "content": LLM_QUESTION}


async def test_llm_timeout_falls_back_without_breaking_history(monkeypatch):
    class Hang(RecordingProvider):
        async def stream(self, messages, *, schema=None, max_tokens=None):
            import asyncio
            await asyncio.sleep(5)
            yield {"type": "token", "text": "x"}

    monkeypatch.setattr(planner, "get_provider", lambda: Hang())
    from app.graph import nodes
    patch_settings(monkeypatch, nodes, llm_deadline_ms=50)
    sid = "g-timeout"
    evs = await run(sid, "10001", LLM_QUESTION)
    assert evs[-1]["data"]["source"] == "llm_timeout" and text(evs) == service.FALLBACK_TIMEOUT
    assert len((await cache.load_state(sid))["turns"]) == 2


async def test_llm_provider_error_is_logged_and_fallback_recommends_retry(monkeypatch, caplog):
    from app.llm import ProviderError

    class BrokenProvider(RecordingProvider):
        async def stream(self, messages, *, schema=None, max_tokens=None):
            raise ProviderError("test provider returned 503")
            yield  # pragma: no cover

    monkeypatch.setattr(planner, "get_provider", lambda: BrokenProvider())
    evs = await run("g-provider-error", "10001", LLM_QUESTION)

    assert evs[-1]["data"]["source"] == "llm_error"
    assert "try again shortly" in text(evs).lower()
    assert "test provider returned 503" in caplog.text


# ---- verification / error branch ----------------------------------------------------------

async def test_unknown_customer_ends_the_graph_after_load_state():
    evs = await run("g-bad", "99999999", "hello")
    assert names(evs) == ["error"] and "No customer found" in evs[0]["data"]["message"]
    assert await cache.session_customer("g-bad") is None


async def test_session_is_bound_to_one_customer():
    sid = "g-bind"
    first = await run(sid, "10001", "What's my customer ID?")
    assert names(first)[-1] == "done"
    again = await run(sid, "", "what is my email?")                   # bound: number no longer needed
    assert names(again) == ["meta", "token", "done"]
    other = await run(sid, "10002", "what is my email?")              # a different customer -> rejected
    assert names(other) == ["error"] and "different customer" in other[0]["data"]["message"]


async def test_guest_session_runs_through_the_graph():
    evs = await run("g-guest", "guest", "hello")
    assert names(evs)[0] == "meta" and names(evs)[-1] == "done"
    assert evs[0]["data"]["customer_id"] is None
