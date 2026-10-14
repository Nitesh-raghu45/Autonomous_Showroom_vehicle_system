"""Text normalization shared by the rule router and the general planner.

This module used to also contain a difflib-based intent guesser (`classify()`)
and, later, an embedding-based semantic router (`semantic.py`, MiniLM-L6-v2).
Both have been removed: anything the rules don't resolve falls through to the
general planner. The normalizer keeps that planner input consistent across
the application (see `rules.route()` and `service.answer()`).

What's left here is a single, narrow job: normalize whitespace and casing for
before any pattern or the LLM sees the text, so "vehicel", "emial",
"shedule" etc. behave like the correctly-spelled word everywhere downstream.
"""
from __future__ import annotations

def normalize(text: str) -> str:
    low = text.lower().replace("’", "'")
    return " ".join(low.split())


def correct(text: str) -> str:
    """Preserve original wording when sending unresolved questions to the planner."""
    return text
