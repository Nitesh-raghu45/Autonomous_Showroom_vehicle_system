"""Conversation history -> LangChain messages, with a token-bounded view for the LLM.

Storage vs. what the model sees
-------------------------------
Redis keeps the COMPLETE, lossless history (``state["turns"]``, a list of ``[role, text]``).
Nothing here ever deletes from it. This module only decides how much of that history is
*sent to the model* on an LLM-path turn, using LangChain message objects:

    all turns --> HumanMessage / AIMessage
              --> last ``recent_turns`` pairs kept verbatim
              --> older messages shortened to ``compact_chars`` characters
              --> ``trim_messages`` enforces a hard ``max_tokens`` cap (drops the oldest first)
              --> if anything had to be dropped, one short note tells the model so

No model call is involved (a background summariser would be a second LLM call), so this costs
microseconds. Short conversations are sent verbatim and unchanged.
"""
from __future__ import annotations

from typing import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, trim_messages

from .config import settings

_ELLIPSIS = "…"
_NOTE_RESERVE_TOKENS = 40       # head-room for the "earlier messages omitted" note so the cap stays hard


def approx_tokens(messages: Sequence[BaseMessage]) -> int:
    """Cheap token estimate (~4 chars/token + per-message overhead). No tokenizer dependency."""
    return sum(len(str(m.content)) // 4 + 4 for m in messages)


def turns_to_messages(turns: Sequence[Sequence[str]]) -> list[BaseMessage]:
    """``[["user", "hi"], ["assistant", "hello"]]`` -> ``[HumanMessage, AIMessage]``."""
    out: list[BaseMessage] = []
    for role, text in turns:
        out.append(HumanMessage(content=text) if role == "user" else AIMessage(content=text))
    return out


def _shorten(msg: BaseMessage, limit: int) -> BaseMessage:
    text = str(msg.content)
    if limit <= 0 or len(text) <= limit:
        return msg
    cut = text[: max(limit - 1, 1)].rstrip() + _ELLIPSIS
    return type(msg)(content=cut)


def build_history(
    turns: Sequence[Sequence[str]],
    *,
    recent_turns: int | None = None,
    compact_chars: int | None = None,
    max_tokens: int | None = None,
) -> list[BaseMessage]:
    """The history the LLM receives for this turn. See the module docstring."""
    recent = settings.llm_history_recent_turns if recent_turns is None else recent_turns
    limit = settings.llm_history_compact_chars if compact_chars is None else compact_chars
    budget = settings.llm_history_max_tokens if max_tokens is None else max_tokens

    messages = turns_to_messages(turns)
    if not messages:
        return []

    keep = max(recent, 0) * 2
    split = max(len(messages) - keep, 0)
    window = [_shorten(m, limit) for m in messages[:split]] + messages[split:]

    if budget <= 0 or approx_tokens(window) <= budget:
        return window

    trimmed = trim_messages(
        window, max_tokens=max(budget - _NOTE_RESERVE_TOKENS, 1), token_counter=approx_tokens,
        strategy="last", start_on="human", include_system=False, allow_partial=False,
    )
    if not trimmed:
        # The most recent exchange alone exceeds the budget: never send a context-free question.
        trimmed = [_shorten(m, limit) for m in window[-2:]]
    dropped = len(window) - len(trimmed)
    if dropped > 0:
        note = SystemMessage(content=(
            f"[{dropped} earlier message(s) in this session were omitted to save space. "
            "The customer JSON above remains the authority for record facts.]"
        ))
        return [note, *trimmed]
    return trimmed
