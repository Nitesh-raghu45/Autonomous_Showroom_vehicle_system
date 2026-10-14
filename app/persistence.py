"""Optional PostgreSQL write-behind (source of truth), OFF unless DATABASE_URL is set.

Redis stays the hot store. Every mutation is enqueued here and flushed by a background
task, so Postgres never sits on the request path. Schema (created on startup):

    entities(kind text, id text, body jsonb, updated_at timestamptz, primary key (kind, id))

EXPERIMENTAL: written against asyncpg's documented API but not exercised in this
repo's test-suite (no Postgres in CI). Run it against staging before trusting it.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from .config import settings

log = logging.getLogger("persistence")

_queue: asyncio.Queue | None = None
_task: asyncio.Task | None = None
_pool = None

_DDL = """
CREATE TABLE IF NOT EXISTS entities (
    kind text NOT NULL, id text NOT NULL, body jsonb NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY (kind, id));
"""
_UPSERT = """
INSERT INTO entities (kind, id, body) VALUES ($1, $2, $3::jsonb)
ON CONFLICT (kind, id) DO UPDATE SET body = EXCLUDED.body, updated_at = now();
"""


def enabled() -> bool:
    return _queue is not None


def enqueue(kind: str, key: str, body: dict[str, Any]) -> None:
    if _queue is not None:
        try:
            _queue.put_nowait((kind, key, json.dumps(body, default=str)))
        except asyncio.QueueFull:
            log.warning("persistence queue full, dropping %s:%s", kind, key)


async def init() -> bool:
    global _queue, _task, _pool
    if not settings.database_url:
        return False
    try:
        import asyncpg  # optional dependency (requirements-postgres.txt)
        _pool = await asyncpg.create_pool(settings.database_url, min_size=1, max_size=4)
        async with _pool.acquire() as conn:
            await conn.execute(_DDL)
    except Exception as exc:  # noqa: BLE001
        log.warning("postgres write-behind disabled: %s", exc)
        return False
    _queue = asyncio.Queue(maxsize=10_000)
    _task = asyncio.create_task(_worker())
    log.info("postgres write-behind enabled")
    return True


async def _worker() -> None:
    assert _queue is not None and _pool is not None
    while True:
        batch = [await _queue.get()]
        while not _queue.empty() and len(batch) < 200:
            batch.append(_queue.get_nowait())
        try:
            async with _pool.acquire() as conn:
                await conn.executemany(_UPSERT, batch)
        except Exception as exc:  # noqa: BLE001
            log.warning("postgres flush failed (%d rows): %s", len(batch), exc)
            await asyncio.sleep(2)
            for item in batch:
                try:
                    _queue.put_nowait(item)
                except asyncio.QueueFull:
                    break


async def close() -> None:
    global _queue
    if _task and _queue is not None:
        while not _queue.empty():
            await asyncio.sleep(0.05)
        _task.cancel()
    if _pool:
        await _pool.close()
    _queue = None
