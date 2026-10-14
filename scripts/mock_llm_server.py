"""Tiny OpenAI-compatible mock LLM (for tests and GPU-less dev).

Streams a canned planner JSON, honouring the same schema the real server would enforce.
Env: MOCK_LLM_DELAY_MS (per token, default 0). Run:  uvicorn scripts.mock_llm_server:app --port 8001
"""
from __future__ import annotations

import asyncio
import json
import os
import re

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI()
DELAY = float(os.getenv("MOCK_LLM_DELAY_MS", "0")) / 1000


def plan_for(question: str) -> dict:
    q = question.lower()
    if "favorite car" in q or "favourite car" in q:
        return {"intent": "REFUSE", "reply": "I can't share favourites, but I can help with your enquiry.", "slots": {}}
    if "book" in q and "callback" in q:
        return {"intent": "BOOK_APPOINTMENT", "reply": "", "slots": {"appointment_type": "CALLBACK", "date": "2026-09-25", "hour": 11, "minute": 0}}
    if "compare" in q:
        return {"intent": "ANSWER", "reply": "The Kia Seltos is cheaper than the BMW X1 in the stock rows.", "slots": {}}
    if "slow" in q:
        return {"intent": "ANSWER", "reply": "slow answer " * 5, "slots": {}}
    return {"intent": "CLARIFY", "reply": "Could you tell me a little more about what you need?", "slots": {}}


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    user = next((m["content"] for m in reversed(body["messages"]) if m["role"] == "user"), "")
    text = json.dumps(plan_for(user))
    n_prompt = sum(len(m["content"]) for m in body["messages"]) // 4

    async def gen():
        for i in range(0, len(text), 6):
            if DELAY:
                await asyncio.sleep(DELAY * 6)
            yield "data: " + json.dumps({"choices": [{"delta": {"content": text[i:i + 6]}}]}) + "\n\n"
        yield "data: " + json.dumps({"choices": [], "usage": {"prompt_tokens": n_prompt, "completion_tokens": len(text) // 4}}) + "\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
