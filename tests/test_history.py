"""LangChain history layer (app/history.py) and the chat-model adapter (app/llm/langchain_model.py)."""
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.history import approx_tokens, build_history, turns_to_messages
from app.llm.langchain_model import ProviderChatModel, to_provider_messages


def _turns(n: int, answer_len: int = 600) -> list[list[str]]:
    out: list[list[str]] = []
    for i in range(1, n + 1):
        out += [["user", f"question number {i}"], ["assistant", f"answer number {i} " + "y" * answer_len]]
    return out


def test_turns_become_langchain_messages():
    msgs = turns_to_messages([["user", "hi"], ["assistant", "hello"]])
    assert isinstance(msgs[0], HumanMessage) and isinstance(msgs[1], AIMessage)
    assert [m.content for m in msgs] == ["hi", "hello"]


def test_short_conversation_is_sent_verbatim():
    turns = _turns(4)
    sent = build_history(turns, recent_turns=6, compact_chars=50, max_tokens=2000)
    assert [m.content for m in sent] == [t[1] for t in turns]


def test_older_messages_are_shortened_recent_stay_verbatim():
    turns = _turns(10)
    sent = build_history(turns, recent_turns=2, compact_chars=40, max_tokens=0)
    assert len(sent) == len(turns)                                   # nothing dropped, only shortened
    assert all(len(m.content) <= 40 for m in sent[:-4])
    assert [m.content for m in sent[-4:]] == [t[1] for t in turns[-4:]]
    assert sent[0].content.startswith("question number 1")           # oldest question still visible


def test_hard_token_cap_drops_oldest_first_and_says_so():
    turns = _turns(60)
    sent = build_history(turns, recent_turns=6, compact_chars=160, max_tokens=800)
    assert approx_tokens(sent) <= 800
    assert isinstance(sent[0], SystemMessage) and "omitted" in sent[0].content
    assert isinstance(sent[1], HumanMessage)                          # never starts mid-exchange
    assert sent[-1].content == turns[-1][1]                           # newest message is always kept


def test_unbounded_mode_sends_everything_verbatim():
    turns = _turns(30)
    sent = build_history(turns, recent_turns=100_000, compact_chars=10, max_tokens=0)
    assert [m.content for m in sent] == [t[1] for t in turns]


def test_oversized_latest_exchange_is_never_dropped_entirely():
    turns = [["user", "q" * 4000], ["assistant", "a" * 4000]]
    sent = build_history(turns, recent_turns=0, compact_chars=100, max_tokens=50)
    assert len(sent) >= 2 and sent[-1].type == "ai"


def test_stored_history_is_never_mutated():
    turns = _turns(40)
    snapshot = [list(t) for t in turns]
    build_history(turns, recent_turns=2, compact_chars=20, max_tokens=300)
    assert turns == snapshot


def test_empty_history():
    assert build_history([]) == []


# ---- chat-model adapter ---------------------------------------------------------------

class FakeProvider:
    name, model = "fake", "fake-1"

    def __init__(self, pieces=("{\"a\":", " 1}")):
        self.pieces, self.calls = pieces, []

    async def stream(self, messages, *, schema=None, max_tokens=None):
        self.calls.append({"messages": messages, "schema": schema, "max_tokens": max_tokens})
        for p in self.pieces:
            yield {"type": "token", "text": p}
        yield {"type": "usage", "usage": {"prompt_tokens": 11, "completion_tokens": 3, "cached_prompt_tokens": 7}, "ttft_ms": 42}


def test_to_provider_messages_maps_roles():
    out = to_provider_messages([SystemMessage(content="s"), HumanMessage(content="h"), AIMessage(content="a")])
    assert out == [{"role": "system", "content": "s"}, {"role": "user", "content": "h"}, {"role": "assistant", "content": "a"}]


@pytest.mark.asyncio
async def test_adapter_streams_text_then_usage_and_forwards_schema():
    prov = FakeProvider()
    model = ProviderChatModel(provider=prov, json_schema={"type": "object"}, max_tokens=50)
    chunks = [c async for c in model.astream([HumanMessage(content="hi")])]
    assert "".join(str(c.content) for c in chunks) == '{"a": 1}'
    # LangChain appends its own empty terminal chunk, so locate ours by content rather than position.
    last = next(c for c in chunks if c.usage_metadata)
    assert last.usage_metadata["input_tokens"] == 11 and last.usage_metadata["output_tokens"] == 3
    assert last.usage_metadata["input_token_details"]["cache_read"] == 7
    assert last.response_metadata["ttft_ms"] == 42
    assert prov.calls == [{"messages": [{"role": "user", "content": "hi"}], "schema": {"type": "object"}, "max_tokens": 50}]


@pytest.mark.asyncio
async def test_adapter_ainvoke_returns_joined_text():
    model = ProviderChatModel(provider=FakeProvider())
    res = await model.ainvoke([HumanMessage(content="hi")])
    assert res.content == '{"a": 1}' and res.usage_metadata["total_tokens"] == 14


# ---- prompt template safety -----------------------------------------------------------------

def test_braces_in_customer_data_questions_or_history_are_never_parsed_as_template_syntax():
    from app.prompting import build_messages

    msgs = build_messages(
        question="what is {customer_id} and {{x}} and {0}?", cid="1",
        record={"customer": {"name": "A {b} {{c}}"}}, appts=[],
        turns=[["user", "earlier {y}"], ["assistant", "reply {z} {{w}}"]],
    )
    assert msgs[-1] == {"role": "user", "content": "what is {customer_id} and {{x}} and {0}?"}
    joined = "\n".join(m["content"] for m in msgs)
    assert "A {b} {{c}}" in joined and "earlier {y}" in joined and "reply {z} {{w}}" in joined
