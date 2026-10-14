"""Available-slot store for the Service Agent (pull / push interface).

Slots are cached in Redis under the ``svc:slots:`` namespace. The store
supports two refresh strategies:
  - **pull**: a background task polls an external service management system.
  - **push**: a webhook endpoint receives slot updates in real time.

When no external system is configured, the store seeds sensible default
slots (every 30-minute interval from 09:00–17:30) so the agent works
out of the box in development.
"""
from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from typing import Any

from ... import cache, clock
from .models import ServiceSlot

log = logging.getLogger("service.slot_store")

# Redis key helpers
_SLOTS_KEY = "svc:slots:{center}:{date}"
_LOCK_KEY = "svc:slot_lock:{center}:{date}:{time}"

# Default dev-mode slot grid
_DEFAULT_CENTER = "default"
_SLOT_START_HOUR = 9
_SLOT_END_HOUR = 18
_SLOT_INTERVAL_MIN = 30
_DEFAULT_CAPACITY = 3                 # vehicles per slot
_DEV_DAYS_AHEAD = 14                  # seed 2 weeks into the future


# ---------------------------------------------------------------------------
# Internal Redis access — mirrors cache module's safety pattern
# ---------------------------------------------------------------------------

def _get_client():
    """Return the shared Redis client from the cache module (or None)."""
    return cache._client


async def _safe(coro, default=None):
    """Execute a Redis coroutine with the same circuit-breaker safety as cache._safe."""
    import time as _time
    client = _get_client()
    if client is None or _time.monotonic() < cache._down_until:
        coro.close()
        return default
    try:
        return await coro
    except Exception as exc:
        log.warning("slot_store redis op failed: %s", exc)
        cache._trip()
        return default


# ---------------------------------------------------------------------------
# Key builders
# ---------------------------------------------------------------------------

def _slots_key(center: str, dt: date) -> str:
    return _SLOTS_KEY.format(center=center, date=dt.isoformat())


def _lock_key(center: str, dt: date, time_str: str) -> str:
    return _LOCK_KEY.format(center=center, date=dt.isoformat(), time=time_str)


# ---------------------------------------------------------------------------
# Seed / default slots
# ---------------------------------------------------------------------------

def _generate_default_slots(center: str, dt: date) -> list[ServiceSlot]:
    """Generate the default 30-minute slot grid for a single day."""
    slots: list[ServiceSlot] = []
    for h in range(_SLOT_START_HOUR, _SLOT_END_HOUR):
        for m in (0, 30):
            time_str = f"{h:02d}:{m:02d}"
            slots.append(ServiceSlot(
                center=center, date=dt, time=time_str,
                capacity=_DEFAULT_CAPACITY, available=_DEFAULT_CAPACITY,
            ))
    return slots


async def seed_default_slots(center: str = _DEFAULT_CENTER) -> int:
    """Seed default slots for the next N days if they don't already exist.

    Returns the number of days seeded (0 if Redis is down or already seeded).
    """
    r = _get_client()
    if r is None:
        log.warning("slot_store.seed: Redis unavailable — skipping")
        return 0

    seeded = 0
    today = clock.today()
    for delta in range(_DEV_DAYS_AHEAD):
        dt = today + timedelta(days=delta)
        key = _slots_key(center, dt)
        exists = await _safe(r.exists(key), 0)
        if exists:
            continue
        slots = _generate_default_slots(center, dt)
        pipe = r.pipeline()
        for s in slots:
            pipe.hset(key, s.time, s.model_dump_json())
        pipe.expire(key, 86400 * (_DEV_DAYS_AHEAD + 1))  # auto-cleanup
        await _safe(pipe.execute())
        seeded += 1
    if seeded:
        log.info("slot_store: seeded %d days of default slots for center=%s", seeded, center)
    return seeded


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

async def get_available(center: str, dt: date) -> list[ServiceSlot]:
    """Return all slots with remaining capacity for the given center+date."""
    r = _get_client()
    if r is None:
        return _generate_default_slots(center, dt)  # fallback for no-Redis dev

    key = _slots_key(center, dt)
    raw: dict = await _safe(r.hgetall(key), {})
    if not raw:
        # Auto-seed if not yet present
        await seed_default_slots(center)
        raw = await _safe(r.hgetall(key), {})

    slots: list[ServiceSlot] = []
    for _time_key, val in raw.items():
        try:
            v = val if isinstance(val, str) else val.decode()
            s = ServiceSlot.model_validate_json(v)
            if s.is_available:
                slots.append(s)
        except Exception:
            log.warning("slot_store: corrupt slot data for %s / %s", key, _time_key)
    return sorted(slots, key=lambda s: s.time)


async def get_slot(center: str, dt: date, time_str: str) -> ServiceSlot | None:
    """Return a specific slot, or None if it doesn't exist."""
    r = _get_client()
    if r is None:
        return None
    key = _slots_key(center, dt)
    raw = await _safe(r.hget(key, time_str))
    if raw is None:
        return None
    v = raw if isinstance(raw, str) else raw.decode()
    return ServiceSlot.model_validate_json(v)


# ---------------------------------------------------------------------------
# Claim / release (atomic)
# ---------------------------------------------------------------------------

async def claim_slot(center: str, dt: date, time_str: str, appt_id: str) -> bool:
    """Atomically decrement ``available`` for the slot.

    Returns True if the claim succeeded, False if the slot is full or missing.
    Uses a Redis lock key to prevent double-booking race conditions.
    """
    r = _get_client()
    if r is None:
        return True           # optimistic in dev mode

    lock = _lock_key(center, dt, time_str)
    acquired = await _safe(r.set(lock, appt_id, nx=True, ex=300), None)
    if not acquired:
        return False          # another booking is racing on this slot

    try:
        key = _slots_key(center, dt)
        raw = await _safe(r.hget(key, time_str))
        if raw is None:
            return False
        v = raw if isinstance(raw, str) else raw.decode()
        slot = ServiceSlot.model_validate_json(v)
        if slot.available <= 0:
            return False
        slot.available -= 1
        await _safe(r.hset(key, time_str, slot.model_dump_json()))
        return True
    finally:
        await _safe(r.delete(lock))


async def release_slot(center: str, dt: date, time_str: str) -> None:
    """Increment ``available`` for a slot (used on cancellation)."""
    r = _get_client()
    if r is None:
        return

    key = _slots_key(center, dt)
    raw = await _safe(r.hget(key, time_str))
    if raw is None:
        return
    v = raw if isinstance(raw, str) else raw.decode()
    slot = ServiceSlot.model_validate_json(v)
    slot.available = min(slot.available + 1, slot.capacity)
    await _safe(r.hset(key, time_str, slot.model_dump_json()))


# ---------------------------------------------------------------------------
# Push interface (called from webhook endpoint)
# ---------------------------------------------------------------------------

async def push_update(center: str, dt: date, slots: list[ServiceSlot]) -> int:
    """Receive pushed slot updates from an external system.

    Overwrites the entire slot grid for the given center+date.
    Returns the number of slots stored.
    """
    r = _get_client()
    if r is None:
        log.warning("slot_store.push: Redis unavailable — update discarded")
        return 0

    key = _slots_key(center, dt)
    pipe = r.pipeline()
    pipe.delete(key)
    for s in slots:
        pipe.hset(key, s.time, s.model_dump_json())
    pipe.expire(key, 86400 * 7)
    await _safe(pipe.execute())
    log.info("slot_store.push: updated %d slots for %s / %s", len(slots), center, dt.isoformat())
    return len(slots)
