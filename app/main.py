import asyncio
import contextvars
import json
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from . import cache, inventory, mutations, observability, persistence, service, sql_engine, store
from .config import BASE_DIR, settings
from .llm import close_client as close_llm_client
from .models import (
    CustomerRegisterRequest,
    CustomerUpdateRequest,
    ScheduleRequest,
    VehicleType,
)
from .mutations import book_appointment, cancel_appointment, register_customer, update_customer
from .ratelimit import limiter
from .service_main import service_router

# ---------------------------------------------------------------------------
# Logging
#
# Every log line carries a request_id so a single HTTP/SSE request can be
# grep'd end-to-end across every module it touches (service, router, planner,
# llm client). The request_id is generated per-request in `request_context`
# below, threaded via a contextvar (no need to pass it through every function
# signature), and echoed back to the browser as `X-Request-Id` so a client-side
# failure (e.g. a raw "Failed to fetch") can be matched to server-side logs by
# the same ID -- or its *absence* from the logs proves the request never
# reached this process at all (dropped by a proxy, or the process itself
# restarted mid-request, e.g. `uvicorn --reload` picking up a saved file).
# ---------------------------------------------------------------------------

_request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id_ctx.get()
        return True


_MOBILE_RE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?([6-9]\d{5})(\d{4})(?!\d)")
_EMAIL_RE = re.compile(r"\b([A-Za-z0-9._%+-])[A-Za-z0-9._%+-]*@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")


class _PiiRedactingFormatter(logging.Formatter):
    """Mask mobile numbers and emails in every log line, whichever module logged it
    (questions, customer_number, slots and LLM outcomes all pass through here)."""

    def format(self, record: logging.LogRecord) -> str:
        out = super().format(record)
        out = _MOBILE_RE.sub(lambda m: "******" + m.group(2), out)
        return _EMAIL_RE.sub(r"\1***@\2", out)


_handler = logging.StreamHandler()
_handler.setFormatter(_PiiRedactingFormatter(
    "%(asctime)s %(levelname)-7s %(name)-10s [rid=%(request_id)s] %(message)s"
))
# The filter must live on the HANDLER, not the logger: a logger's own filters
# only run when you log directly to that logger. Every other module's logger
# (service, router, third-party libs like httpx) reaches this handler via
# propagation and bypasses per-logger filters entirely -- so attaching it to
# the logger left every non-"api"-logger record missing `request_id` and
# raised a KeyError inside the formatter itself.
_handler.addFilter(_RequestIdFilter())
logging.basicConfig(level=logging.INFO, handlers=[_handler])
log = logging.getLogger("api")


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

PROCESS_INSTANCE_ID = uuid.uuid4().hex[:8]
PROCESS_STARTED_AT = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Logged once per worker boot. If this line appears again mid-session for the
    # same client (i.e. between two requests that share a session_id), the worker
    # restarted -- the #1 cause of an in-flight request dying with a bare
    # "Failed to fetch" in the browser (uvicorn --reload picking up a saved file,
    # a crash, a container restart, or an orchestrator health-check kill). Grep
    # for "worker boot" and compare its timestamp to the failed request's time.
    log.warning("worker boot: instance_id=%s pid=%s", PROCESS_INSTANCE_ID, __import__("os").getpid())
    count = store.load(settings.data_file)
    log.info("loaded %d customer records from %s", count, settings.data_file)

    redis_ok = await cache.init()
    log.info(
        "provider=%s model=%s redis=%s",
        settings.llm_provider, settings.model_name, "up" if redis_ok else "down",
    )

    await store.seed_redis()                       # no-op when already seeded
    await cache.migrate_legacy_appointments()      # one-time, guarded by a flag key
    inv_count = await inventory.init()             # in-memory indexed catalogue
    log.info("inventory ready: %d vehicles", inv_count)
    sql_ok = sql_engine.init(inventory.catalog().items)  # in-memory SQLite for analytical queries
    if sql_ok:
        log.info("SQL engine ready")
    await persistence.init()                       # optional Postgres write-behind (DATABASE_URL)

    # Seed default service appointment slots (dev mode, no-op if already seeded)
    from .agents.service.slot_store import seed_default_slots
    await seed_default_slots()
    warm = asyncio.create_task(service.warm_llm()) # connection + prefix-cache warm-up, off the boot path

    yield

    warm.cancel()
    await sql_engine.close()
    await persistence.close()
    await cache.close()
    await close_llm_client()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Customer Assist",
    version="2.0.0",
    description="Vehicle dealership CRM assistant with self-service portal",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
# Without this middleware `default_limits` is never applied and every undecorated
# route (inventory, telemetry, the /service router) is unlimited.
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-Id"],
)


@app.middleware("http")
async def request_id_and_access_log(request: Request, call_next):
    """Assign/propagate a request id and log the full lifecycle of every request.

    This is what makes a client-reported "Failed to fetch" diagnosable after the
    fact: the browser sends the id it generated for that attempt back as
    `X-Request-Id` (see frontend/src/api.js), so a client-error report can be
    joined against this access log by id. Three outcomes are distinguished:
      - a normal response (status + duration logged)
      - the client disconnecting before we finished writing (very common for
        SSE: the person navigated away, or their network dropped) -- NOT a
        server bug, logged as "client disconnected" rather than an exception
      - an unhandled exception escaping the route entirely, which previously
        just showed up in the browser as a bare "Failed to fetch" with nothing
        server-side to correlate it to
    """
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    token = _request_id_ctx.set(rid)
    t0 = time.perf_counter()
    log.info("request start: method=%s path=%s client=%s", request.method, request.url.path, request.client.host if request.client else "?")
    try:
        response = await call_next(request)
        response.headers["X-Request-Id"] = rid
        log.info("request end: path=%s status=%s duration_ms=%d", request.url.path, response.status_code, int((time.perf_counter() - t0) * 1000))
        return response
    except asyncio.CancelledError:
        # The client (or an intermediate proxy) closed the connection before we
        # finished responding. For a streaming endpoint this is the expected
        # shape of "user closed the tab" or "their wifi dropped" -- it is NOT
        # the server crashing, and should not be confused with the exception
        # branch below when triaging an incident.
        log.warning("client disconnected mid-request: path=%s duration_ms=%d", request.url.path, int((time.perf_counter() - t0) * 1000))
        raise
    except Exception:
        log.exception("unhandled exception in request: path=%s duration_ms=%d", request.url.path, int((time.perf_counter() - t0) * 1000))
        observability.bump("unhandled_exception_total")
        raise
    finally:
        _request_id_ctx.reset(token)


# ---------------------------------------------------------------------------
# Request/response schemas (chat only - others live in models.py)
# ---------------------------------------------------------------------------

class SessionStart(BaseModel):
    customer_number: str = Field(..., description="Customer ID or mobile number")


class ChatRequest(BaseModel):
    session_id:      Optional[str] = None
    customer_number: str = ""          # only needed for first verification; afterwards session_id is sufficient
    question:        str = Field(..., min_length=1, max_length=2000)


class ClientErrorReport(BaseModel):
    """What the browser saw when a fetch/stream failed on its side.

    This is the piece that was completely missing before: when `fetch()` itself
    throws (network refused, DNS failure, connection reset mid-stream), that
    failure happens entirely in the browser and the server never sees a request
    at all -- so nothing about it ever reached the server logs. The frontend now
    POSTs a best-effort report here (see reportClientError in api.js) so these
    incidents are visible and queryable instead of vanishing the moment the tab
    is closed or the person moves on.
    """
    request_id:      Optional[str] = None
    session_id:      Optional[str] = None
    customer_number: Optional[str] = None
    endpoint:        str
    kind:            str            # "fetch_error" | "stream_read_error" | "unhandled_rejection"
    message:         str
    navigator_online: Optional[bool] = None
    elapsed_ms:      Optional[int] = None
    user_agent:      Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _resolve_or_404(number: str) -> str:
    cid = await cache.resolve_number(number)
    if cid is None:
        raise HTTPException(404, f"No customer found for '{number}'")
    return cid


# ---------------------------------------------------------------------------
# System routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["system"])
async def health():
    """Liveness + dependency check."""
    return {
        "status":            "ok",
        "customers_loaded":  len(store.all_ids()),
        "redis":             "up" if await cache.ping() else "down",
        "provider":          settings.llm_provider,
        "model":             settings.model_name,
        "router":            settings.router_enabled,
        "postgres":          persistence.enabled(),
        "inventory_total":   inventory.count_inventory()["total"],
        "intent_classifier": {"enabled": False, "ready": False, "backend": None, "model": None, "error": None},
        "agents":            {"sales": "active", "service": "active", "management": "stub"},
    }


@app.get("/metrics", tags=["system"])
async def metrics():
    """Per-worker latency percentiles by route (router / llm / ...)."""
    return observability.snapshot()


@app.get("/metrics/prometheus", tags=["system"], response_class=PlainTextResponse)
async def metrics_prometheus():
    return observability.prometheus()


# ---------------------------------------------------------------------------
# Customer read routes
# ---------------------------------------------------------------------------

@app.get("/customers", tags=["customers"])
@limiter.limit("10/minute")
async def list_customers(request: Request):
    return {"count": len(store.all_ids()), "customers": store.summaries()}


@app.get("/customers/{number}", tags=["customers"])
@limiter.limit("30/minute")
async def get_customer(request: Request, number: str):
    cid = await _resolve_or_404(number)
    record, src = await cache.get_customer_record_ex(cid)
    return {
        "customer_id":    cid,
        "record":         record,
        "profile_block":  store.render_record(record, cid) if record else None,
        "profile_cache":  src,
    }


# ---------------------------------------------------------------------------
# Customer mutation routes
# ---------------------------------------------------------------------------

@app.post("/customer/register", tags=["customers"], status_code=201)
@limiter.limit("10/minute")
async def register_new_customer(request: Request, body: CustomerRegisterRequest):
    """Register a brand-new customer. Returns the new customer ID."""
    try:
        result = await register_customer(body)
    except mutations.Conflict as exc:
        raise HTTPException(409, str(exc))
    except cache.StoreUnavailable as exc:
        raise HTTPException(503, str(exc))
    return result


@app.patch("/customer/{customer_id}/info", tags=["customers"])
@limiter.limit("20/minute")
async def update_customer_info(request: Request, customer_id: str, body: CustomerUpdateRequest):
    """Self-service update of name, email, or alternate mobile."""
    try:
        result = await update_customer(customer_id, body)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    except mutations.Conflict as exc:
        raise HTTPException(409, str(exc))
    except cache.StoreUnavailable as exc:
        raise HTTPException(503, str(exc))
    return result


# ---------------------------------------------------------------------------
# Scheduling routes
# ---------------------------------------------------------------------------

@app.post("/customer/{customer_id}/schedule", tags=["appointments"], status_code=201)
@limiter.limit("20/minute")
async def book_schedule(request: Request, customer_id: str, body: ScheduleRequest):
    """Book a new appointment (test drive, callback, advance payment, etc.)."""
    cid = await _resolve_or_404(customer_id)
    try:
        result = await book_appointment(cid, body)
    except mutations.Conflict as exc:
        raise HTTPException(409, str(exc))
    except cache.StoreUnavailable as exc:
        raise HTTPException(503, str(exc))
    return result


@app.get("/customer/{customer_id}/schedules", tags=["appointments"])
@limiter.limit("30/minute")
async def list_schedules(request: Request, customer_id: str):
    """List all appointments for a customer."""
    cid = await _resolve_or_404(customer_id)
    appointments = await cache.list_appointments(cid)
    return {"customer_id": cid, "appointments": appointments, "count": len(appointments)}


@app.patch("/customer/{customer_id}/schedule/{appt_id}/cancel", tags=["appointments"])
@limiter.limit("20/minute")
async def cancel_schedule(request: Request, customer_id: str, appt_id: str):
    """Cancel an existing appointment."""
    cid = await _resolve_or_404(customer_id)
    try:
        result = await cancel_appointment(cid, appt_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    except cache.StoreUnavailable as exc:
        raise HTTPException(503, str(exc))
    return result


# ---------------------------------------------------------------------------
# Session routes
# ---------------------------------------------------------------------------

@app.post("/session/start", tags=["session"])
@limiter.limit("20/minute")
async def start_session(request: Request, body: SessionStart):
    cid = await cache.resolve_number(body.customer_number)
    if cid is None:
        raise HTTPException(404, f"No customer found for '{body.customer_number}'")

    session_id = uuid.uuid4().hex
    await cache.bind_session(session_id, cid)
    record = await cache.get_customer_record(cid) or {}
    return {
        "session_id":    session_id,
        "verified":      True,
        "customer_id":   cid,
        "customer_name": (record.get("customer") or {}).get("name"),
        "ttl_seconds":   settings.session_ttl,
        # The full record stays server-side and is supplied to both the rule
        # engine and the LLM. It is intentionally not returned to the browser.
    }


@app.delete("/session/{session_id}", tags=["session"])
@limiter.limit("20/minute")
async def end_session(request: Request, session_id: str):
    await cache.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}


# ---------------------------------------------------------------------------
# Chat routes
# ---------------------------------------------------------------------------

HEARTBEAT_INTERVAL_S = 15.0  # keep intermediate proxies (nginx, load balancers) from
                             # closing an SSE connection during a slow LLM turn; also
                             # gives us a log line proving the server was still alive
                             # and producing output, which rules out a server-side hang
                             # as the cause of a client-visible "Failed to fetch".


@app.post("/chat/stream", tags=["chat"])
@limiter.limit("60/minute")
async def chat_stream(request: Request, body: ChatRequest):
    """SSE stream: meta -> token* -> [action] -> done | error."""
    session_id = body.session_id or uuid.uuid4().hex
    t_open = time.perf_counter()
    log.info("chat stream open: session_id=%s customer_number=%s question=%r", session_id, body.customer_number, body.question)

    async def generator() -> AsyncIterator[str]:
        yield _sse("session", {"session_id": session_id})
        last_emit = time.perf_counter()
        next_chunk: asyncio.Task | None = None
        try:
            inner = service.answer(session_id, body.customer_number, body.question).__aiter__()
            next_chunk = asyncio.create_task(inner.__anext__())
            while True:
                completed, _ = await asyncio.wait({next_chunk}, timeout=HEARTBEAT_INTERVAL_S)
                if not completed:
                    if await request.is_disconnected():
                        log.warning(
                            "client disconnected during LLM wait: session_id=%s elapsed_ms=%d",
                            session_id, int((time.perf_counter() - t_open) * 1000),
                        )
                        return
                    idle_s = time.perf_counter() - last_emit
                    log.info("chat stream heartbeat: session_id=%s idle_s=%.1f (still waiting on upstream)", session_id, idle_s)
                    yield ": heartbeat\n\n"
                    continue
                try:
                    chunk = next_chunk.result()
                except StopAsyncIteration:
                    next_chunk = None
                    break
                next_chunk = asyncio.create_task(inner.__anext__())
                last_emit = time.perf_counter()
                yield _sse(chunk["event"], chunk["data"])
            log.info("chat stream complete: session_id=%s total_ms=%d", session_id, int((time.perf_counter() - t_open) * 1000))
        except asyncio.CancelledError:
            log.warning(
                "chat stream cancelled (client closed connection): session_id=%s elapsed_ms=%d",
                session_id, int((time.perf_counter() - t_open) * 1000),
            )
            raise
        except Exception:
            log.exception("chat stream failed: session_id=%s elapsed_ms=%d", session_id, int((time.perf_counter() - t_open) * 1000))
            observability.bump("stream_exception_total")
            yield _sse("error", {"message": "stream failed on server"})
        finally:
            if next_chunk is not None and not next_chunk.done():
                next_chunk.cancel()
                await asyncio.gather(next_chunk, return_exceptions=True)
        yield "event: end\ndata: {}\n\n"

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection":    "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/telemetry/client-error", tags=["telemetry"])
@limiter.limit("30/minute")
async def client_error(request: Request, report: ClientErrorReport):
    """Receive a best-effort report of a client-side network failure.

    Fire-and-forget from the browser (see reportClientError in api.js) - it must
    never throw or block the UI, and its delivery is not guaranteed (if the
    person's network is down, this call can fail too). But when it *does* land,
    it turns an otherwise invisible "Failed to fetch" into a queryable, timestamped,
    request_id-correlated log line and counter.
    """
    log.warning(
        "client-reported network error: request_id=%s session_id=%s endpoint=%s kind=%s "
        "message=%r online=%s elapsed_ms=%s ua=%r",
        report.request_id, report.session_id, report.endpoint, report.kind,
        report.message, report.navigator_online, report.elapsed_ms, report.user_agent,
    )
    observability.bump(f"client_error_total|{report.kind}")
    return {"status": "logged"}


@app.post("/chat", tags=["chat"])
@limiter.limit("60/minute")
async def chat_once(request: Request, body: ChatRequest):
    """Non-streaming variant."""
    session_id = body.session_id or uuid.uuid4().hex
    log.info("chat request start: session_id=%s customer_number=%s question=%r", session_id, body.customer_number, body.question)
    parts, meta, done, error, action = [], {}, {}, None, None
    try:
        async for chunk in service.answer(session_id, body.customer_number, body.question):
            if chunk["event"] == "token":
                parts.append(chunk["data"]["text"])
            elif chunk["event"] == "replace":
                parts = [chunk["data"]["text"]]
            elif chunk["event"] == "meta":
                meta = chunk["data"]
            elif chunk["event"] == "done":
                done = chunk["data"]
            elif chunk["event"] == "action":
                action = chunk["data"]
            elif chunk["event"] == "error":
                error = chunk["data"]["message"]
    except Exception:
        log.exception("chat request failed: session_id=%s customer_number=%s", session_id, body.customer_number)
        raise
    if error:
        log.warning("chat request error: session_id=%s error=%s", session_id, error)
        raise HTTPException(502, error)
    body_out = {
        "session_id": session_id,
        "answer":     "".join(parts),
        "meta":       meta,
        "stats":      done,
        "action":     action,
    }
    t = (done or {}).get("timings", {})
    st = ", ".join(f"{k.replace('_ms', '')};dur={v}" for k, v in t.items())
    log.info("chat request complete: session_id=%s source=%s route=%s latency_ms=%s", session_id, (done or {}).get("source"), (done or {}).get("route"), (done or {}).get("latency_ms"))
    return JSONResponse(body_out, headers={"Server-Timing": st})


# ---------------------------------------------------------------------------
# Inventory routes
# ---------------------------------------------------------------------------

@app.get("/inventory/makes", tags=["inventory"])
async def get_makes():
    """List all vehicle makes in stock."""
    makes = inventory.list_makes()
    return {"makes": makes, "count": len(makes)}


@app.get("/inventory/count", tags=["inventory"])
async def get_inventory_count():
    return inventory.count_inventory()


@app.get("/inventory/search", tags=["inventory"])
async def search_inventory(
    make:         Optional[str] = None,       # comma-separated for several makes
    model:        Optional[str] = None,
    type:         VehicleType = VehicleType.ALL,
    fuel:         Optional[str] = None,
    transmission: Optional[str] = None,
    body:         Optional[str] = None,
    min_price:    Optional[int] = None,
    max_price:    Optional[int] = None,
    max_km:       Optional[int] = None,
    sort:         Optional[str] = None,
    limit:        Optional[int] = None,
):
    """Search inventory (in-memory indexed - no Redis on this path)."""
    makes = [m.strip() for m in make.split(",")] if make else None
    total, rows = inventory.catalog().search(
        makes=makes, models=[model] if model else None, vtype=type.value, fuel=fuel,
        transmission=transmission, body=body, min_price=min_price, max_price=max_price,
        max_km=max_km, sort=sort, limit=limit,
    )
    return {"results": rows, "count": len(rows), "total": total}


@app.get("/inventory/{make}/models", tags=["inventory"])
async def get_models(make: str):
    """List all models for a given make."""
    models = inventory.list_models(make)
    return {"make": make, "models": models, "count": len(models)}


# ---------------------------------------------------------------------------
# Service Agent REST routes
# ---------------------------------------------------------------------------

app.include_router(service_router, prefix="/service", tags=["service"])


# ---------------------------------------------------------------------------
# Static / SPA
# ---------------------------------------------------------------------------

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

FRONTEND_DIST = BASE_DIR / "frontend" / "dist"
if (FRONTEND_DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")


@app.get("/", include_in_schema=False)
@app.get("/{full_path:path}", include_in_schema=False)
async def spa_fallback(full_path: str = ""):
    """Serve React SPA for all unmatched routes (client-side routing)."""
    built = FRONTEND_DIST / "index.html"
    if built.is_file():
        return FileResponse(built)
    return FileResponse(BASE_DIR / "static" / "index.html")