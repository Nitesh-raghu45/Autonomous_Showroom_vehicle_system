"""Ollama streaming client (/api/chat, newline-delimited JSON). Development only.

Ollama is convenient but is not a production inference server (no continuous batching
or prefix-aware scheduling). Use vLLM / SGLang (`LLM_PROVIDER=local`) in production.
Fixes vs. the old version: shared keep-alive client, JSON-schema `format`, small `num_predict`.
"""
from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator

from ..config import settings
from .base import ProviderError, empty_usage
from .http import get_client


class OllamaProvider:
    name = "ollama"

    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        self.model = model or settings.ollama_model
        self._base_url = (base_url or settings.ollama_base_url).rstrip("/")

    async def stream(self, messages, *, schema=None, max_tokens=None, timeout: float | None = None) -> AsyncIterator[dict[str, Any]]:
        payload: dict[str, Any] = {
            "model": self.model, "messages": messages, "stream": True, "keep_alive": "30m",
            "options": {"temperature": settings.temperature, "num_predict": max_tokens or settings.llm_max_tokens, "num_ctx": 4096},
        }
        if schema is not None and settings.structured_mode != "none":
            payload["format"] = schema if settings.structured_mode == "json_schema" else "json"
        usage = empty_usage()
        t0 = time.perf_counter()
        ttft: int | None = None
        approx = sum(len(m["content"]) for m in messages) // 4
        async with get_client().stream("POST", f"{self._base_url}/api/chat", json=payload, timeout=timeout) as resp:
            if resp.status_code >= 400:
                raise ProviderError(f"ollama returned {resp.status_code}: {(await resp.aread()).decode(errors='replace')[:300]}")
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                piece = (chunk.get("message") or {}).get("content")
                if piece:
                    if ttft is None:
                        ttft = int((time.perf_counter() - t0) * 1000)
                    yield {"type": "token", "text": piece}
                if chunk.get("done"):
                    processed = int(chunk.get("prompt_eval_count") or 0)
                    usage["prompt_tokens"] = processed or approx
                    usage["completion_tokens"] = int(chunk.get("eval_count") or 0)
                    if processed and approx > processed:
                        usage["cached_prompt_tokens"] = approx - processed
        yield {"type": "usage", "usage": usage, "ttft_ms": ttft or 0}

    async def warmup(self, messages) -> None:
        try:
            async for _ in self.stream(messages, max_tokens=1):
                pass
        except Exception:  # noqa: BLE001
            pass
