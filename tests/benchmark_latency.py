"""Latency + accuracy benchmark against a RUNNING server, driven by the 200-query dataset.

    python -m tests.benchmark_latency --base-url http://localhost:8000 \
        --concurrency 1,5,10,25,50 --repeat 3 --out results/

Reports P50/P90/P95/P99/max, % of requests under the SLA, TTFT, per-route breakdown,
error rate, LLM token usage and objective accuracy (see tests/scoring.py) for every
concurrency level. Mutating categories are excluded by default so repeated runs stay
comparable; add --include-mutations to exercise them once at concurrency 1.

Start the server with RATE_LIMIT_ENABLED=false (and TODAY_OVERRIDE=2026-09-18 if you
want the dataset's calendar to line up).
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import httpx

try:
    from tests.scoring import score
except ModuleNotFoundError:  # run as a plain script
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tests.scoring import score

MUTATING = {"customer_registration", "update_customer_info", "book_appointment", "cancel_appointment"}


def pct(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    return round(s[min(len(s) - 1, max(0, int(round(p / 100 * (len(s) - 1)))))], 1)


async def one(client: httpx.AsyncClient, q: dict, base: str) -> dict:
    body = {"customer_number": q["customer_id"] or "", "question": q["query"]}
    t0 = time.perf_counter()
    ttft = None
    parts: list[str] = []
    action = done = None
    err = None
    try:
        async with client.stream("POST", f"{base}/chat/stream", json=body, timeout=30) as r:
            if r.status_code != 200:
                err = f"HTTP {r.status_code}"
            event = None
            async for line in r.aiter_lines():
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:") and event:
                    data = json.loads(line[5:])
                    if event == "token":
                        if ttft is None:
                            ttft = (time.perf_counter() - t0) * 1000
                        parts.append(data["text"])
                    elif event == "replace":
                        parts = [data["text"]]
                    elif event == "action":
                        action = data
                    elif event == "done":
                        done = data
                    elif event == "error":
                        err = data.get("message")
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
    total = (time.perf_counter() - t0) * 1000
    answer = "".join(parts)
    ok, why = (None, "error") if err else score(q, answer, action)
    return {
        "id": q["id"], "category": q["category"], "query": q["query"], "answer": answer, "error": err,
        "total_ms": total, "ttft_ms": ttft, "route": (done or {}).get("route", "error"),
        "source": (done or {}).get("source"), "server_ms": (done or {}).get("latency_ms"),
        "completion_tokens": ((done or {}).get("usage") or {}).get("completion_tokens", 0),
        "prompt_tokens": ((done or {}).get("usage") or {}).get("prompt_tokens", 0),
        "scored": ok is not None, "correct": bool(ok), "why": why,
    }


async def run_level(base: str, queries: list[dict], conc: int) -> tuple[list[dict], float]:
    sem = asyncio.Semaphore(conc)
    limits = httpx.Limits(max_connections=conc + 5, max_keepalive_connections=conc + 5)
    async with httpx.AsyncClient(limits=limits) as client:
        async def guarded(q):
            async with sem:
                return await one(client, q, base)
        t0 = time.perf_counter()
        res = await asyncio.gather(*(guarded(q) for q in queries))
        return list(res), time.perf_counter() - t0


def summarise(rows: list[dict], wall: float, sla: float) -> dict:
    ok_rows = [r for r in rows if not r["error"]]
    lat = [r["total_ms"] for r in ok_rows]
    by_route: dict[str, list[float]] = defaultdict(list)
    for r in ok_rows:
        by_route[r["route"]].append(r["total_ms"])
    scored = [r for r in rows if r["scored"]]
    return {
        "n": len(rows), "errors": len(rows) - len(ok_rows), "rps": round(len(rows) / wall, 1),
        "p50": pct(lat, 50), "p90": pct(lat, 90), "p95": pct(lat, 95), "p99": pct(lat, 99), "max": round(max(lat), 1) if lat else 0,
        "mean": round(statistics.mean(lat), 1) if lat else 0,
        "under_sla_pct": round(100 * sum(1 for v in lat if v <= sla) / len(lat), 1) if lat else 0,
        "ttft_p95": pct([r["ttft_ms"] for r in ok_rows if r["ttft_ms"] is not None], 95),
        "routes": {k: {"n": len(v), "p50": pct(v, 50), "p95": pct(v, 95), "p99": pct(v, 99)} for k, v in by_route.items()},
        "llm_share_pct": round(100 * len(by_route.get("llm", [])) / len(ok_rows), 1) if ok_rows else 0,
        "avg_completion_tokens": round(statistics.mean([r["completion_tokens"] for r in ok_rows]), 1) if ok_rows else 0,
        "accuracy": {"scored": len(scored), "correct": sum(r["correct"] for r in scored),
                     "pct": round(100 * sum(r["correct"] for r in scored) / len(scored), 1) if scored else None},
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--dataset", default="data/teknnova_benchmark_dataset.json")
    ap.add_argument("--concurrency", default="1,5,10,25")
    ap.add_argument("--repeat", type=int, default=1, help="repeat the dataset N times per level")
    ap.add_argument("--sla-ms", type=float, default=500)
    ap.add_argument("--include-mutations", action="store_true")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    data = json.load(open(a.dataset))
    queries = [q for q in data if a.include_mutations or q["category"] not in MUTATING]
    print(f"{len(queries)} queries x {a.repeat} repeat(s); mutating categories {'included' if a.include_mutations else 'excluded'}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient() as c:      # warm-up: one request per route type, results discarded
        for q in queries[:5]:
            await one(c, q, a.base_url)

    summary = {}
    header = f"{'conc':>4} {'n':>5} {'err':>4} {'rps':>6} {'p50':>7} {'p90':>7} {'p95':>7} {'p99':>7} {'max':>8} {'<SLA%':>6} {'llm%':>5} {'acc%':>6}"
    print(header)
    all_rows = []
    for conc in [int(x) for x in a.concurrency.split(",")]:
        rows, wall = await run_level(a.base_url, queries * a.repeat, conc)
        s = summarise(rows, wall, a.sla_ms)
        summary[conc] = s
        all_rows += [{**r, "concurrency": conc} for r in rows]
        print(f"{conc:>4} {s['n']:>5} {s['errors']:>4} {s['rps']:>6} {s['p50']:>7} {s['p90']:>7} {s['p95']:>7} {s['p99']:>7} {s['max']:>8} "
              f"{s['under_sla_pct']:>6} {s['llm_share_pct']:>5} {s['accuracy']['pct']}")
        for route, v in s["routes"].items():
            print(f"       {route:<14} n={v['n']:<5} p50={v['p50']:<7} p95={v['p95']:<7} p99={v['p99']}")
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    with open(out / "rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    wrong = [r for r in all_rows if r["scored"] and not r["correct"] and r["concurrency"] == all_rows[0]["concurrency"]]
    if wrong:
        print("\nIncorrect (first concurrency level):")
        for r in wrong:
            print(f"  {r['id']} {r['query'][:70]!r} -> {r['why']}")
    print(f"\nwrote {out/'summary.json'} and {out/'rows.csv'}")


if __name__ == "__main__":
    asyncio.run(main())
