"""The LangGraph workflow - deliberately tiny (5 nodes, no LLM node except the fallback):

        START
          |
     load_state ----(error)----> END
          |
        route            (deterministic router)
        /    \\
  rule_handler  llm_handler      (rule path = no model; LLM path = ONE LangChain call)
        \\    /
     save_history
          |
         END

Compiled once at import time of first use and shared by every request (compiled graphs are
safe for concurrent runs). No checkpointer: the session (customer binding + complete history +
pending action) already lives in Redis and is loaded by `load_state` in a single pipelined round
trip. A LangGraph checkpointer would add a Redis write per node and require Redis Stack.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, AsyncIterator

from langgraph.graph import END, StateGraph

from .nodes import after_load, llm_handler, load_state, route_decision, route_question, rule_handler, save_history
from .state import ChatState


def build_graph():
    g = StateGraph(ChatState)
    g.add_node("load_state", load_state)
    g.add_node("route", route_question)
    g.add_node("rule_handler", rule_handler)
    g.add_node("llm_handler", llm_handler)
    g.add_node("save_history", save_history)

    g.set_entry_point("load_state")
    g.add_conditional_edges("load_state", after_load, {"route": "route", "end": END})
    g.add_conditional_edges("route", route_decision, {"rule": "rule_handler", "llm": "llm_handler"})
    g.add_edge("rule_handler", "save_history")
    g.add_edge("llm_handler", "save_history")
    g.add_edge("save_history", END)
    return g.compile()


@lru_cache(maxsize=1)
def get_graph():
    return build_graph()


async def run_chat(session_id: str, customer_number: str, question: str) -> AsyncIterator[dict[str, Any]]:
    """Run one turn and yield the SSE-style events (`meta`, `token`, `replace`, `action`, `done` | `error`)."""
    async for event in get_graph().astream(
        {"session_id": session_id, "customer_number": customer_number, "question": question},
        stream_mode="custom",
    ):
        yield event
