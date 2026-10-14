"""In-process latency accounting: rolling percentiles per route + Prometheus text.

No Redis writes on the request path (the old code did several sequential HINCRBYs per
chat turn). Prometheus scrapes each worker; percentiles here are per worker.
"""
from __future__ import annotations

import bisect
import time
from collections import defaultdict, deque

_BUCKETS_MS = [5, 10, 25, 50, 100, 150, 250, 350, 500, 750, 1000, 2000, 5000]
_WINDOW = 5000

_started = time.time()
_lat: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=_WINDOW))
_hist: dict[str, list[int]] = defaultdict(lambda: [0] * (len(_BUCKETS_MS) + 1))
_sum: dict[str, float] = defaultdict(float)
_counters: dict[str, int] = defaultdict(int)


def record(route: str, total_ms: float) -> None:
    _lat[route].append(total_ms)
    _lat["all"].append(total_ms)
    for r in (route, "all"):
        _hist[r][bisect.bisect_left(_BUCKETS_MS, total_ms)] += 1
        _sum[r] += total_ms
    _counters[f"requests_total|{route}"] += 1
    if total_ms > 500:
        _counters["sla_breach_total"] += 1


def bump(name: str, n: int = 1) -> None:
    _counters[name] += n


def _pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = min(len(sorted_vals) - 1, max(0, int(round(p / 100 * (len(sorted_vals) - 1)))))
    return round(sorted_vals[k], 1)


def snapshot() -> dict:
    routes = {}
    for name, vals in _lat.items():
        s = sorted(vals)
        routes[name] = {
            "count": len(s), "p50_ms": _pct(s, 50), "p90_ms": _pct(s, 90),
            "p95_ms": _pct(s, 95), "p99_ms": _pct(s, 99), "max_ms": round(s[-1], 1) if s else 0,
            "under_500ms_pct": round(100 * sum(1 for v in s if v <= 500) / len(s), 1) if s else 0,
        }
    return {"uptime_s": int(time.time() - _started), "latency": routes, "counters": dict(_counters)}


def prometheus() -> str:
    out = ["# TYPE chat_latency_ms histogram"]
    for route, buckets in _hist.items():
        cum = 0
        for edge, n in zip(_BUCKETS_MS, buckets):
            cum += n
            out.append(f'chat_latency_ms_bucket{{route="{route}",le="{edge}"}} {cum}')
        cum += buckets[-1]
        out.append(f'chat_latency_ms_bucket{{route="{route}",le="+Inf"}} {cum}')
        out.append(f'chat_latency_ms_sum{{route="{route}"}} {_sum[route]:.1f}')
        out.append(f'chat_latency_ms_count{{route="{route}"}} {cum}')
    out.append("# TYPE chat_counter counter")
    for k, v in _counters.items():
        out.append(f'chat_counter{{name="{k}"}} {v}')
    return "\n".join(out) + "\n"
