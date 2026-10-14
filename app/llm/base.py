"""Provider interface.

Every provider is an async generator that yields dicts:

    {"type": "token", "text": "..."}                       zero or more, in order
    {"type": "usage", "usage": {...}, "ttft_ms": int}      exactly one, last

`schema` (optional) is a JSON Schema the server should enforce with constrained
decoding. `usage` is normalised to {prompt_tokens, completion_tokens, cached_prompt_tokens}.
"""
from __future__ import annotations

from typing import Any, AsyncIterator, Protocol


class LLMProvider(Protocol):
    name: str
    model: str

    def stream(
        self,
        messages: list[dict[str, str]],
        *,
        schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        ...

    async def warmup(self, messages: list[dict[str, str]]) -> None:
        ...


def empty_usage() -> dict[str, int]:
    return {"prompt_tokens": 0, "completion_tokens": 0, "cached_prompt_tokens": 0}


class ProviderError(RuntimeError):
    pass
