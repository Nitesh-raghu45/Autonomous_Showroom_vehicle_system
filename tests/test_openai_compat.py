from __future__ import annotations

import dataclasses

import httpx
import pytest

from app.llm import openai_compat


@pytest.mark.asyncio
async def test_openai_compatible_provider_sends_bearer_auth_and_retries_transient_errors(monkeypatch):
    seen_authorization: list[str | None] = []
    responses = iter((503, 200))

    def respond(request: httpx.Request) -> httpx.Response:
        seen_authorization.append(request.headers.get("Authorization"))
        status = next(responses)
        if status == 503:
            return httpx.Response(status, headers={"Retry-After": "0"}, text="temporarily unavailable")
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=(
                b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
                b'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":1}}\n\n'
                b"data: [DONE]\n\n"
            ),
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(openai_compat, "get_client", lambda: client)
    monkeypatch.setattr(openai_compat, "settings", dataclasses.replace(openai_compat.settings, llm_max_retries=1))
    provider = openai_compat.OpenAICompatProvider(
        "test-provider", "https://llm.example/v1", "test-secret", "test-model"
    )

    try:
        events = [
            event async for event in provider.stream(
                [{"role": "user", "content": "ping"}], max_tokens=10
            )
        ]
    finally:
        await client.aclose()

    assert seen_authorization == ["Bearer test-secret", "Bearer test-secret"]
    assert [event["text"] for event in events if event["type"] == "token"] == ["ok"]
    assert events[-1]["type"] == "usage"
    assert events[-1]["usage"]["prompt_tokens"] == 12

