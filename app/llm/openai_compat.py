"""Streaming client for any OpenAI-compatible /chat/completions endpoint.

Used for the self-hosted vLLM / SGLang server (production), and for Groq / OpenAI.

Latency notes
-------------
* Shared keep-alive client (llm/http.py) - no per-request handshake.
* ``response_format`` enforces the planner JSON schema with constrained decoding
  (xgrammar/outlines server-side), so the reply is always parseable.
* If a server rejects json_schema (HTTP 400) we downgrade once to json_object and
  remember it - no user-visible failure.
* Retries are deliberately tiny: a slow retry is worse than the deterministic fallback.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator

from ..config import settings
from .base import ProviderError, empty_usage
from .http import get_client

log = logging.getLogger("llm")


def _retry_delay(attempt: int, retry_after: str | None = None) -> float:
    try:
        delay = float(retry_after) if retry_after else 0.1 * (2 ** attempt)
    except ValueError:
        delay = 0.1 * (2 ** attempt)
    return min(max(delay, 0.1), 2.0)


class OpenAICompatProvider:
    def __init__(self, name: str, base_url: str, api_key: str, model: str, *, key_required: bool = True):
        self.name = name
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._mode = settings.structured_mode
        if key_required and not api_key:
            raise ProviderError(f"{name.upper()}_API_KEY is not set. Add it to .env or switch LLM_PROVIDER.")

    def _payload(self, messages, schema, max_tokens, attempt: int = 0) -> dict[str, Any]:
        budget = max_tokens or settings.llm_max_tokens
        if attempt > 0 and max_tokens is None:
            # gpt-oss style reasoning models spend completion tokens on hidden reasoning; a retry with the
            # identical budget at temperature 0 just reproduces the same empty reply, so give it more room.
            budget *= 2
        p: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": settings.temperature,
            "max_tokens": budget,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if settings.llm_reasoning_effort and self.name in ("groq", "openai"):
            p["reasoning_effort"] = settings.llm_reasoning_effort
        if schema is not None:
            if self._mode == "json_schema":
                p["response_format"] = {"type": "json_schema", "json_schema": {"name": "plan", "schema": schema}}
            elif self._mode == "json_object":
                p["response_format"] = {"type": "json_object"}
        return p

    @staticmethod
    def _usage(raw: dict[str, Any]) -> dict[str, int]:
        u = empty_usage()
        block = raw.get("usage") or (raw.get("x_groq") or {}).get("usage") or {}
        if block:
            u["prompt_tokens"] = int(block.get("prompt_tokens") or 0)
            u["completion_tokens"] = int(block.get("completion_tokens") or 0)
            u["cached_prompt_tokens"] = int((block.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        return u

    async def stream(self, messages, *, schema=None, max_tokens=None, timeout: float | None = None) -> AsyncIterator[dict[str, Any]]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        url = f"{self._base_url}/chat/completions"
        usage = empty_usage()
        ttft_ms: int | None = None
        t0 = time.perf_counter()

        for attempt in range(1 + settings.llm_max_retries):
            started_tokens = False
            try:
                async with get_client().stream("POST", url, headers=headers, json=self._payload(messages, schema, max_tokens, attempt), timeout=timeout) as resp:
                    if resp.status_code >= 400:
                        body = (await resp.aread()).decode(errors="replace")[:400]
                        if resp.status_code == 400 and schema is not None and self._mode == "json_schema":
                            log.warning("%s rejected json_schema (%s) - downgrading to json_object", self.name, body[:120])
                            self._mode = "json_object"
                            continue
                        if (resp.status_code in (408, 425, 429) or 500 <= resp.status_code < 600) and attempt < settings.llm_max_retries:
                            await asyncio.sleep(_retry_delay(attempt, resp.headers.get("Retry-After")))
                            continue
                        raise ProviderError(f"{self.name} returned {resp.status_code}: {body}")
                    finish_reason = None
                    stream_error = None
                    saw_reasoning = False
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        if chunk.get("error"):
                            # Groq can answer HTTP 200 and then put the failure inside the stream.
                            stream_error = chunk["error"]
                            log.warning("%s in-stream error: %s", self.name, str(stream_error)[:300])
                            # Rescue json_validate_failed: the model produced valid text but
                            # not in JSON format.  Wrap it so the planner can use it.
                            if isinstance(stream_error, dict) and stream_error.get("code") == "json_validate_failed":
                                rescued = (stream_error.get("failed_generation") or "").strip()
                                if rescued:
                                    wrapped = json.dumps({"intent": "CLARIFY", "reply": rescued})
                                    if ttft_ms is None:
                                        ttft_ms = int((time.perf_counter() - t0) * 1000)
                                    started_tokens = True
                                    yield {"type": "token", "text": wrapped}
                            continue
                        found = self._usage(chunk)
                        if found["prompt_tokens"] or found["completion_tokens"]:
                            usage = found
                        for choice in chunk.get("choices") or []:
                            delta = choice.get("delta") or {}
                            if delta.get("reasoning"):
                                saw_reasoning = True
                            piece = delta.get("content")
                            if piece:
                                if ttft_ms is None:
                                    ttft_ms = int((time.perf_counter() - t0) * 1000)
                                started_tokens = True
                                yield {"type": "token", "text": piece}
                            if choice.get("finish_reason"):
                                finish_reason = choice["finish_reason"]
                    # If the model returned a valid response but zero content tokens,
                    # retry once - Groq's openai/gpt-oss-20b occasionally does this.
                    if not started_tokens:
                        log.warning("%s returned zero content tokens (finish_reason=%s, reasoning_seen=%s, stream_error=%s, usage=%s) - attempt %d",
                                    self.name, finish_reason, saw_reasoning, str(stream_error)[:200] if stream_error else None, usage, attempt + 1)
                        if attempt < settings.llm_max_retries:
                            await asyncio.sleep(_retry_delay(attempt))
                            continue
                        # Out of retries: fail loudly HERE so planner does not fire 2 more identical calls.
                        raise ProviderError(f"{self.name} returned no content (finish_reason={finish_reason}, error={stream_error})")
                break
            except ProviderError:
                raise
            except (asyncio.CancelledError, GeneratorExit):
                raise
            except Exception as exc:  # noqa: BLE001  (connect errors etc.)
                if started_tokens or attempt >= settings.llm_max_retries:
                    raise ProviderError(f"{self.name} request failed: {exc}") from exc
                await asyncio.sleep(_retry_delay(attempt))
        yield {"type": "usage", "usage": usage, "ttft_ms": ttft_ms or 0}

    async def warmup(self, messages) -> None:
        """Open the TCP/TLS connection and prime the server's prefix cache."""
        try:
            async for _ in self.stream(messages, max_tokens=8):
                pass
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM warm-up failed (server not up yet?): %s", exc)


def local() -> OpenAICompatProvider:
    return OpenAICompatProvider(
        "local",
        getattr(settings, "local_base_url", settings.ollama_base_url),
        getattr(settings, "local_api_key", ""),
        getattr(settings, "local_model", settings.ollama_model),
        key_required=False,
    )


def groq() -> OpenAICompatProvider:
    return OpenAICompatProvider("groq", settings.groq_base_url, settings.groq_api_key, settings.groq_model)


def openai() -> OpenAICompatProvider:
    return OpenAICompatProvider(
        "openai",
        getattr(settings, "openai_base_url", settings.groq_base_url),
        getattr(settings, "openai_api_key", settings.groq_api_key),
        getattr(settings, "openai_model", settings.groq_model),
    )