"""One process-wide keep-alive HTTP client for every LLM provider.

The old Ollama provider built a fresh ``httpx.AsyncClient`` (new TCP, and for cloud
providers new TLS) for every request. Reusing one pooled client removes that handshake
from the request path.
"""
from __future__ import annotations

import httpx

from ..config import settings

_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=settings.llm_connect_timeout_s,
                read=max(settings.llm_deadline_ms / 1000 * 2, 10.0),
                write=5.0,
                pool=5.0,
            ),
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=100, keepalive_expiry=120),
        )
    return _client


async def close_client() -> None:
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None
