"""Redis layer (L2) + in-process cache (L1).

Hot-path rules this module enforces
-----------------------------------
* ONE Redis round trip per chat turn: ``fetch_bundle`` pipelines the session state,
  the customer record (only when L1 misses) and the appointment hash (only when the
  routed intent needs it).
* Customer records and number->id lookups are cached in process (L1). Other workers
  are told to drop an entry through a Redis pub/sub channel when a record changes.
* Session state is Redis-first, never read from L1 while Redis is reachable (requests
  are not sticky to a worker, so a worker-local cache would silently diverge from what
  other workers see). Its write-back is fire-and-forget and the next read waits for it
  if still in flight. When Redis is unreachable, session state falls back to a small
  in-process TTL cache (keyed by session id) so a single instance still remembers a
  conversation instead of silently forgetting it every turn - this fallback is
  best-effort only and is NOT shared across worker processes.
* Registration is a single atomic Lua script (duplicate check + id allocation +
  record + indexes) instead of six sequential commands.
* A tiny circuit breaker stops a dead Redis from adding its timeout to every request.

Key layout
----------
  customer:{id}              JSON record (mutable, Redis-first, JSON file is the seed)
  customer:mobile:{num}      reverse lookup mobile -> id
  customer:all               set of ids
  customers:next_id          INCR counter
  appts:{id}                 HASH appointment_id -> JSON       (was: N+1 GETs)
  sst:{session}              JSON compact session state        (was: raw history list)
  inv:stock                  list of JSON vehicles (startup only)
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any

import redis.asyncio as aioredis

from .config import settings
from .l1 import TTLCache

log = logging.getLogger("cache")

_client: aioredis.Redis | None = None
_listener_client: aioredis.Redis | None = None
_worker_id = uuid.uuid4().hex[:8]
_listener: asyncio.Task | None = None
_down_until = 0.0

_records = TTLCache(settings.l1_max_entries, settings.l1_ttl_s)     # cid -> record
_numbers = TTLCache(settings.l1_max_entries, 60)                    # normalised number/id -> cid
_session_l1 = TTLCache(settings.l1_max_entries, settings.session_ttl)  # sid -> state; fallback ONLY (see below)
_inflight: dict[str, asyncio.Task] = {}                              # session_id -> write-behind task

INVALIDATION_CHANNEL = "teknnova:invalidate"


class StoreUnavailable(RuntimeError):
    """Raised by mutations when Redis (the durable store) cannot be reached."""


# ---------------------------------------------------------------------------
# Lua
# ---------------------------------------------------------------------------

_REGISTER_LUA = """
-- KEYS[1] = customer:mobile:<mobile>   KEYS[2] = customers:next_id   KEYS[3] = customer:all
-- ARGV[1] = alt mobile ('' if none)    ARGV[2] = record JSON with __CID__ placeholder
local existing = redis.call('GET', KEYS[1])
if existing then return {0, existing} end
if ARGV[1] ~= '' then
  local e2 = redis.call('GET', 'customer:mobile:' .. ARGV[1])
  if e2 then return {2, e2} end
end
local id = tostring(redis.call('INCR', KEYS[2]))
local rec = (string.gsub(ARGV[2], '__CID__', id))
redis.call('SET', 'customer:' .. id, rec)
redis.call('SET', KEYS[1], id)
if ARGV[1] ~= '' then redis.call('SET', 'customer:mobile:' .. ARGV[1], id) end
redis.call('SADD', KEYS[3], id)
return {1, id}
"""

_CAS_LUA = """
local cur = redis.call('GET', KEYS[1])
if (cur == false and ARGV[1] == '') or cur == ARGV[1] then
  redis.call('SET', KEYS[1], ARGV[2])
  return 1
end
return 0
"""

_register_script = None
_cas_script = None


# ---------------------------------------------------------------------------
# lifecycle
# ---------------------------------------------------------------------------

async def init() -> bool:
    global _client, _listener_client, _register_script, _cas_script, _listener
    try:
        if settings.redis_url.startswith("fakeredis"):
            from fakeredis import FakeAsyncRedis          # dev/test only (requirements-dev.txt)
            _client = FakeAsyncRedis(decode_responses=True)
            _listener_client = _client
        else:
            # Cap the pool size to avoid exceeding Upstash (or small Redis)
            # connection limits, especially during --reload restarts.
            pool_cap = min(settings.redis_pool_size, 10)
            _client = aioredis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=settings.redis_timeout_ms / 1000,
                socket_keepalive=True,
                max_connections=pool_cap,
                health_check_interval=30,
            )
            # Pub/Sub waits indefinitely for invalidation messages. It needs
            # socket_timeout=None so it doesn't break after the command timeout.
            # max_connections=1 keeps total connections within Upstash limits.
            _listener_client = aioredis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=None,
                socket_keepalive=True,
                max_connections=1,
            )
        await _client.ping()
        _register_script = _client.register_script(_REGISTER_LUA)
        _cas_script = _client.register_script(_CAS_LUA)
        _listener = asyncio.create_task(_invalidation_listener())
        log.info("redis connected")
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("redis unavailable - reads fall back to JSON seed, writes disabled: %s", exc)
        _client = None
        return False


async def close() -> None:
    global _listener, _listener_client
    if _listener:
        _listener.cancel()
        try:
            await _listener        # let the CancelledError actually unwind and
        except asyncio.CancelledError:  # release ps.aclose() above, BEFORE we
            pass                    # force-close the connection pool below.
        _listener = None
    for t in list(_inflight.values()):
        try:
            await t
        except Exception:  # noqa: BLE001
            pass
    if _client is not None:
        await _client.aclose()
    if _listener_client is not None and _listener_client is not _client:
        await _listener_client.aclose()
    _listener_client = None


def available() -> bool:
    return _client is not None and time.monotonic() >= _down_until


def _trip() -> None:
    """Open the circuit for a few seconds after a Redis failure."""
    global _down_until
    _down_until = time.monotonic() + 5.0


async def _safe(coro, default=None):
    if _client is None or time.monotonic() < _down_until:
        coro.close()
        return default
    try:
        return await coro
    except Exception as exc:  # noqa: BLE001
        log.warning("redis op failed: %s", exc)
        _trip()
        return default


async def ping() -> bool:
    return bool(await _safe(_client.ping() if _client else _noop(), False))


async def _noop():
    return None


# ---------------------------------------------------------------------------
# cross-worker L1 invalidation
# ---------------------------------------------------------------------------

async def _publish(cid: str) -> None:
    if _client is not None:
        await _safe(_client.publish(INVALIDATION_CHANNEL, f"{_worker_id}:{cid}"))


async def _invalidation_listener() -> None:
    while True:
        try:
            assert _listener_client is not None
            ps = _listener_client.pubsub()
            try:
                await ps.subscribe(INVALIDATION_CHANNEL)
                async for msg in ps.listen():
                    if msg.get("type") != "message":
                        continue
                    worker, _, cid = str(msg["data"]).partition(":")
                    if worker != _worker_id:
                        _records.pop(cid)
                        _numbers.clear()
            finally:
                # release the subscriber connection deterministically - without this,
                # cancelling the task on shutdown leaves it mid-read on a socket with
                # no timeout, and the pool's disconnect() has to force-close it later.
                await ps.aclose()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.warning("invalidation listener restarting: %s", exc)
            await asyncio.sleep(2)


# ---------------------------------------------------------------------------
# customer records
# ---------------------------------------------------------------------------

def _rk(cid: str) -> str:
    return f"customer:{cid}"


def peek_record(cid: str) -> dict[str, Any] | None:
    """L1-only read (no I/O)."""
    return _records.get(cid)


async def get_customer_record_ex(cid: str) -> tuple[dict[str, Any] | None, str]:
    """Return (record, source) where source is l1 | redis | json | none."""
    rec = _records.get(cid)
    if rec is not None:
        return rec, "l1"
    raw = await _safe(_client.get(_rk(cid))) if _client else None
    if raw:
        try:
            rec = json.loads(raw)
            _records.set(cid, rec)
            return rec, "redis"
        except json.JSONDecodeError:
            pass
    from . import store
    rec = store.get(cid)
    return rec, ("json" if rec else "none")


async def get_customer_record(cid: str) -> dict[str, Any] | None:
    return (await get_customer_record_ex(cid))[0]


async def set_customer_record(cid: str, record: dict[str, Any]) -> None:
    if _client is None:
        raise StoreUnavailable("Redis is not available")
    ok = await _safe(_client.set(_rk(cid), json.dumps(record)), None)
    if ok is None:
        raise StoreUnavailable("Redis write failed")
    _records.set(cid, record)
    await _publish(cid)


async def update_customer_record(cid: str, fn) -> dict[str, Any]:
    """Optimistic read-modify-write. ``fn(record) -> new_record`` must be pure."""
    if _client is None:
        raise StoreUnavailable("Redis is not available")
    from . import store
    for _ in range(4):
        raw = await _safe(_client.get(_rk(cid)), "__err__")
        if raw == "__err__":
            raise StoreUnavailable("Redis read failed")
        record = json.loads(raw) if raw else store.get(cid)
        if record is None:
            raise KeyError(f"Customer {cid} not found")
        new = fn(record)
        ok = await _safe(_cas_script(keys=[_rk(cid)], args=[raw or "", json.dumps(new)]), None)
        if ok == 1:
            _records.set(cid, new)
            await _publish(cid)
            return new
        if ok is None:
            raise StoreUnavailable("Redis write failed")
    raise StoreUnavailable("Concurrent update conflict, please retry")


async def register_customer_atomic(record_template: dict[str, Any], mobile: str, alt: str | None) -> tuple[str, str]:
    """Atomically create a customer. Returns (status, id).

    status: "created" | "duplicate_mobile" | "duplicate_alt"; id is the new id, or the
    id that already owns the number.
    """
    if _client is None:
        raise StoreUnavailable("Redis is not available")
    res = await _safe(
        _register_script(
            keys=[f"customer:mobile:{mobile}", "customers:next_id", "customer:all"],
            args=[alt or "", json.dumps(record_template)],
        ),
        None,
    )
    if res is None:
        raise StoreUnavailable("Redis write failed")
    code, cid = int(res[0]), str(res[1])
    if code == 1:
        _numbers.clear()
        await _publish(cid)
        return "created", cid
    return ("duplicate_mobile" if code == 0 else "duplicate_alt"), cid


def remember_customer(cid: str, record: dict[str, Any]) -> None:
    """Seed the local cache immediately after a successful registration."""
    _records.set(cid, record)
    customer = record.get("customer") or {}
    for value in (customer.get("mobile_no"), customer.get("alt_mobile_no")):
        if value:
            _numbers.set(str(value), cid)


# ---------------------------------------------------------------------------
# id / mobile resolution
# ---------------------------------------------------------------------------

async def set_mobile_lookup(mobile: str, cid: str) -> None:
    if _client is not None:
        await _safe(_client.set(f"customer:mobile:{mobile}", cid))
    _numbers.set(mobile, cid)


async def claim_mobile(mobile: str, cid: str) -> bool:
    """Atomically reserve a number for ``cid`` (SET NX). True if reserved now or already ours."""
    if _client is None:
        raise StoreUnavailable("Redis is not available")
    ok = await _safe(_client.set(f"customer:mobile:{mobile}", cid, nx=True), "__err__")
    if ok == "__err__":
        raise StoreUnavailable("Redis write failed")
    if ok:
        return True
    return (await _safe(_client.get(f"customer:mobile:{mobile}"))) == cid


async def release_mobile(mobile: str, cid: str) -> None:
    """Drop a number's lookup, but only if it still points at ``cid``."""
    if _client is not None and mobile:
        if await _safe(_client.get(f"customer:mobile:{mobile}")) == cid:
            await _safe(_client.delete(f"customer:mobile:{mobile}"))
    _numbers.clear()


async def get_mobile_lookup(mobile: str) -> str | None:
    hit = _numbers.get(mobile)
    if hit:
        return hit
    cid = await _safe(_client.get(f"customer:mobile:{mobile}")) if _client else None
    if cid:
        _numbers.set(mobile, cid)
    return cid


async def resolve_number(user_input: str) -> str | None:
    """Accept a customer id or a mobile number; L1 first, Redis second, JSON seed last."""
    from . import store
    raw = (user_input or "").strip()
    if not raw:
        return None
    hit = _numbers.get(raw)
    if hit:
        return hit
    mobile = store.normalise_mobile(raw)
    cid: str | None = None
    if len(mobile) == 10:
        cid = await get_mobile_lookup(mobile)
    if not cid and (_client is None or time.monotonic() < _down_until):
        cid = store.resolve_id(raw)          # JSON seed only when Redis is down (else stale numbers would still log in)
    if not cid and _records.get(raw) is not None:
        cid = raw
    if not cid and raw.isdigit() and _client is not None:
        if await _safe(_client.exists(_rk(raw)), 0):
            cid = raw
    if cid:
        _numbers.set(raw, cid)
    return cid


# ---------------------------------------------------------------------------
# appointments (one HASH per customer -> one HGETALL)
# ---------------------------------------------------------------------------

def _sorted_appts(mapping: dict[str, str] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for raw in (mapping or {}).values():
        try:
            out.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    out.sort(key=lambda a: a.get("created_at", ""), reverse=True)
    return out


async def save_appointment(cid: str, appt: dict[str, Any]) -> None:
    if _client is None:
        raise StoreUnavailable("Redis is not available")
    ok = await _safe(_client.hset(f"appts:{cid}", appt["appointment_id"], json.dumps(appt)), None)
    if ok is None:
        raise StoreUnavailable("Redis write failed")


async def claim_slot(cid: str, date: str, time_: str, appt_id: str) -> bool:
    """Atomic 'one active appointment per customer per date+time' guard (race-proof, unlike a read-then-write check)."""
    if _client is None:
        raise StoreUnavailable("Redis is not available")
    ok = await _safe(_client.set(f"slot:{cid}:{date}:{time_}", appt_id, nx=True), "__err__")
    if ok == "__err__":
        raise StoreUnavailable("Redis write failed")
    return bool(ok)


async def release_slot(cid: str, date: str, time_: str, appt_id: str) -> None:
    if _client is not None and await _safe(_client.get(f"slot:{cid}:{date}:{time_}")) == appt_id:
        await _safe(_client.delete(f"slot:{cid}:{date}:{time_}"))


# Check-and-add in one round trip so two concurrent bookings can't both take the last seat.
_CLAIM_CAPACITY_LUA = """
if redis.call('SISMEMBER', KEYS[1], ARGV[1]) == 1 then return 1 end
if redis.call('SCARD', KEYS[1]) >= tonumber(ARGV[2]) then return 0 end
redis.call('SADD', KEYS[1], ARGV[1])
return 1
"""


async def claim_capacity(appt_type: str, date: str, time_: str, appt_id: str, capacity: int) -> bool:
    """Showroom-wide guard: at most `capacity` active appointments of this type per date+time,
    across ALL customers (claim_slot only stops one customer double-booking themselves)."""
    if _client is None:
        raise StoreUnavailable("Redis is not available")
    ok = await _safe(_client.eval(_CLAIM_CAPACITY_LUA, 1, f"cap:{appt_type}:{date}:{time_}", appt_id, capacity), "__err__")
    if ok == "__err__":
        raise StoreUnavailable("Redis write failed")
    return bool(int(ok))


async def release_capacity(appt_type: str, date: str, time_: str, appt_id: str) -> None:
    if _client is not None:
        await _safe(_client.srem(f"cap:{appt_type}:{date}:{time_}", appt_id))


async def get_appointment(cid: str, appt_id: str) -> dict[str, Any] | None:
    raw = await _safe(_client.hget(f"appts:{cid}", appt_id)) if _client else None
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


async def list_appointments(cid: str) -> list[dict[str, Any]]:
    mapping = await _safe(_client.hgetall(f"appts:{cid}"), {}) if _client else {}
    return _sorted_appts(mapping)


async def migrate_legacy_appointments() -> int:
    """One-time: move old ``appt:{cid}:{id}`` keys into the ``appts:{cid}`` hash."""
    if _client is None or await _safe(_client.get("migrated:appts_v2")):
        return 0
    moved = 0
    try:
        async for key in _client.scan_iter(match="appt:*:*", count=500):
            parts = key.split(":")
            if len(parts) != 3 or parts[1] == "list":
                continue
            raw = await _client.get(key)
            if raw:
                try:
                    a = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                a.setdefault("created_at", "0000")
                await _client.hset(f"appts:{parts[1]}", parts[2], json.dumps(a))
                moved += 1
        await _client.set("migrated:appts_v2", "1")
    except Exception as exc:  # noqa: BLE001
        log.warning("legacy appointment migration skipped: %s", exc)
    if moved:
        log.info("migrated %d legacy appointments", moved)
    return moved


# ---------------------------------------------------------------------------
# session state (compact, replaces raw history + runtime summarisation)
# ---------------------------------------------------------------------------

def _sk(sid: str) -> str:
    return f"sst:{sid}"


def new_state(cid: str | None = None) -> dict[str, Any]:
    return {"cid": cid, "turns": [], "pending": None, "ctx": {}}


def _parse_state(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        st = json.loads(raw)
        return st if isinstance(st, dict) else None
    except json.JSONDecodeError:
        return None


async def _await_inflight(sid: str) -> None:
    t = _inflight.get(sid)
    if t is not None:
        try:
            await t
        except Exception:  # noqa: BLE001
            pass


async def bind_session(sid: str, cid: str) -> None:
    await save_state(sid, new_state(cid))


async def load_state(sid: str) -> dict[str, Any] | None:
    await _await_inflight(sid)
    return _parse_state(await _safe(_client.get(_sk(sid))) if _client else None)


async def save_state(sid: str, state: dict[str, Any]) -> None:
    # Keep the complete conversational history so relative references like "same date"
    # are still resolvable even when the user asks a follow-up after several turns.
    state["turns"] = state.get("turns", [])
    # Always keep the in-process fallback warm - cheap, and it's the only memory a
    # single instance has left the moment Redis drops out mid-conversation.
    _session_l1.set(sid, state)
    if _client:
        await _safe(_client.set(_sk(sid), json.dumps(state, ensure_ascii=False), ex=settings.session_ttl))


def save_state_bg(sid: str, state: dict[str, Any]) -> None:
    """Fire-and-forget write-back (off the request's critical path)."""
    task = asyncio.create_task(save_state(sid, state))
    _inflight[sid] = task
    task.add_done_callback(lambda t, s=sid: _inflight.pop(s, None) if _inflight.get(s) is t else None)


async def clear_session(sid: str) -> None:
    await _await_inflight(sid)
    _session_l1.pop(sid)
    if _client:
        await _safe(_client.delete(_sk(sid)))


async def session_customer(sid: str) -> str | None:
    st = await load_state(sid)
    return (st or {}).get("cid")


async def fetch_bundle(cid: str | None, sid: str, *, want_appts: bool) -> dict[str, Any]:
    """The one Redis round trip of a chat turn."""
    out: dict[str, Any] = {"state": None, "record": None, "record_src": "none", "appts": [], "redis": True}
    rec = _records.get(cid) if cid else None
    if rec is not None:
        out["record"], out["record_src"] = rec, "l1"
    need_rec = bool(cid) and rec is None

    if _client is None or time.monotonic() < _down_until:
        out["redis"] = False
        await _await_inflight(sid)            # don't race the previous turn's fire-and-forget save
        out["state"] = _session_l1.get(sid)   # best-effort memory while Redis is down
    else:
        await _await_inflight(sid)
        try:
            pipe = _client.pipeline(transaction=False)
            pipe.get(_sk(sid))
            if need_rec:
                pipe.get(_rk(cid))
            if want_appts and cid:
                pipe.hgetall(f"appts:{cid}")
            res = await pipe.execute()
            i = 0
            out["state"] = _parse_state(res[i]); i += 1
            if need_rec:
                raw = res[i]; i += 1
                if raw:
                    try:
                        out["record"] = json.loads(raw)
                        out["record_src"] = "redis"
                        _records.set(cid, out["record"])
                    except json.JSONDecodeError:
                        pass
            if want_appts and cid:
                out["appts"] = _sorted_appts(res[i])
        except Exception as exc:  # noqa: BLE001
            log.warning("bundle fetch failed: %s", exc)
            _trip()
            out["redis"] = False
            out["state"] = _session_l1.get(sid)   # don't drop the conversation just because this one round trip failed

    if cid and out["record"] is None:
        from . import store
        out["record"] = store.get(cid)
        out["record_src"] = "json" if out["record"] else "none"
    return out


# ---------------------------------------------------------------------------
# seeding / startup helpers
# ---------------------------------------------------------------------------

async def seed_customers(by_id: dict[str, dict[str, Any]], normalise) -> None:
    if _client is None:
        return
    if await _safe(_client.exists("customer:all"), 0):
        log.info("redis already seeded, skipping customer seed")
        return
    log.info("seeding %d customer records into redis ...", len(by_id))
    max_id = 10000
    try:
        pipe = _client.pipeline(transaction=False)
        for cid, rec in by_id.items():
            pipe.set(_rk(cid), json.dumps(rec))
            pipe.sadd("customer:all", cid)
            cust = rec.get("customer") or {}
            for key in ("mobile_no", "alt_mobile_no"):
                num = cust.get(key)
                if num:
                    pipe.set(f"customer:mobile:{normalise(str(num))}", cid)
            try:
                max_id = max(max_id, int(cid))
            except ValueError:
                pass
        pipe.set("customers:next_id", max_id, nx=True)   # INCR hands out max_id+1 next
        await pipe.execute()
    except Exception as exc:  # noqa: BLE001
        log.warning("customer seed failed: %s", exc)


async def seed_inventory(items: list[dict[str, Any]]) -> None:
    """Seed inv:stock from the JSON fallback. Only called by inventory.init() when
    load_inventory() has already found the key empty - but that check and this write
    were previously two separate round trips (EXISTS, then a loop of RPUSH). If the
    EXISTS call alone hit a transient failure, _safe() swallowed it to a falsy default
    and this function pushed a second full copy on top of an already-seeded list,
    permanently duplicating every vehicle in Redis. A DEL immediately before the pushes,
    in the same atomic pipeline, makes this self-correcting instead of order-dependent:
    whatever was there (nothing, or a stale duplicated list) is replaced in one round trip.
    """
    if _client is None:
        return
    try:
        pipe = _client.pipeline(transaction=True)
        pipe.delete("inv:stock")
        for item in items:
            pipe.rpush("inv:stock", json.dumps(item))
        await pipe.execute()
    except Exception as exc:  # noqa: BLE001
        log.warning("inventory seed failed: %s", exc)


async def load_inventory() -> list[dict[str, Any]]:
    raw = await _safe(_client.lrange("inv:stock", 0, -1), []) if _client else []
    out = []
    for item in raw or []:
        try:
            out.append(json.loads(item))
        except json.JSONDecodeError:
            continue
    return out