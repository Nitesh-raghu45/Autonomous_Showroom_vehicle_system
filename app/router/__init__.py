"""Deterministic intent router (the 'fast path' decision).

`route(question, ctx)` returns a `Route` in microseconds using rules + vocabulary
lookups - no model, no I/O. Anything it is not confident about is sent to the LLM
planner (`Route.intent == "LLM"`).
"""
from .intents import Route, RouteContext  # noqa: F401
from .rules import route  # noqa: F401
