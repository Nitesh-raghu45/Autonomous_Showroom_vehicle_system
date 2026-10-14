"""Show how each benchmark category is routed and print every LLM-bound query.

    python -m scripts.router_report
"""
import asyncio
import collections
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
os.environ.setdefault("TODAY_OVERRIDE", "2026-09-18")


async def main() -> None:
    from app import inventory, store
    from app.router import RouteContext, route

    root = pathlib.Path(__file__).resolve().parent.parent
    store.load(root / "data" / "customers.json")
    await inventory.init()
    data = json.load(open(root / "data" / "teknnova_benchmark_dataset.json"))
    per = collections.defaultdict(collections.Counter)
    llm = []
    for q in data:
        cid = q["customer_id"]
        name = ((store.get(cid) or {}).get("customer") or {}).get("name") if cid else None
        r = route(q["query"], RouteContext(customer_bound=bool(cid), customer_id=cid, customer_name=name))
        per[q["category"]][r.intent] += 1
        if r.intent == "LLM":
            llm.append((q["id"], q["query"], r.reason))
    for cat, c in per.items():
        print(f"{cat:<28} {dict(c)}")
    print(f"\nLLM-bound: {len(llm)}/{len(data)} ({100 * len(llm) / len(data):.1f}%)")
    for i, q, why in llm:
        print(f"  {i}  {q[:80]}  [{why}]")


asyncio.run(main())
