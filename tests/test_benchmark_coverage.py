"""Runs the full 200-query dataset in-process (fakeredis + mock LLM) and enforces:
   * >= 90 % of queries are served by the deterministic router (no model call)
   * every objectively-scored deterministic answer is correct
This is a regression net for the router, NOT a measurement of LLM quality.
"""
import json
from pathlib import Path

import httpx
import pytest
import pytest_asyncio

from tests.scoring import score

pytestmark = pytest.mark.asyncio(loop_scope="module")
DATA = json.load(open(Path(__file__).resolve().parent.parent / "data" / "teknnova_benchmark_dataset.json"))


@pytest_asyncio.fixture(scope="module")
async def results():
    from app.llm import http as llm_http
    from app.main import app
    from scripts import mock_llm_server
    llm_http._client = httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_llm_server.app), base_url="http://mock-llm")
    out = []
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
            for q in DATA:
                r = await c.post("/chat", json={"customer_number": q["customer_id"] or "", "question": q["query"]})
                d = r.json()
                out.append((q, d))
    return out


async def test_router_coverage(results):
    llm = [q["id"] for q, d in results if d["stats"]["route"] == "llm"]
    assert len(llm) / len(results) <= 0.10, f"too many LLM-routed queries: {llm}"


async def test_scored_accuracy(results):
    bad = []
    n = 0
    for q, d in results:
        if d["stats"]["route"] != "router":
            continue
        ok, why = score(q, d["answer"], d["action"])
        if ok is None:
            continue
        n += 1
        if not ok:
            bad.append((q["id"], q["query"][:60], why))
    assert n >= 90
    assert not bad, bad


async def test_deterministic_latency(results):
    lat = sorted(d["stats"]["latency_ms"] for q, d in results if d["stats"]["route"] == "router")
    assert lat[int(len(lat) * 0.95)] < 100, "router-path P95 should be far below 100 ms in-process"
