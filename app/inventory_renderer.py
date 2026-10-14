"""Deterministic answer text for inventory results. Templates only - no model.

Plain text on purpose: the chat UI renders `white-space: pre-wrap`, not markdown.
"""
from __future__ import annotations

from typing import Any

from . import inventory_query as Q
from . import render
from .config import settings


def lakh(v: float) -> str:
    return render.lakh(int(round(v)))


def vline(v: dict[str, Any]) -> str:
    km = f", {v['mileage_km']:,} km" if v["type"] == "used" else ""
    return (
        f"- {v['year']} {v['make']} {v['model']} {v['variant']} - {v['type']}, "
        f"{v['fuel']}, {v['transmission']}, {lakh(v['price'])}, {v['color']}{km} "
        f"[{v['stock_id']}]"
    )


def describe(f: dict[str, Any], n: int | None = None) -> str:
    """Human description of a filter set: 'new automatic 7-seater suvs under ₹30.00L'."""
    bits: list[str] = []
    if f.get("vtype") in ("new", "used"):
        bits.append(f["vtype"])
    if f.get("colors"):
        bits.append("/".join(f["colors"]))
    fuels = [f["fuel"]] if f.get("fuel") else list(f.get("fuels") or [])
    if fuels:
        bits.append("/".join(x if x == "CNG" else x.lower() for x in fuels))
    if f.get("transmission"):
        bits.append(f["transmission"].lower())
    if f.get("min_seats"):
        bits.append(f"{f['min_seats']}-seater")
    makes = f.get("makes") or []
    if f.get("models"):
        names = " / ".join((f"{makes[0]} {m}" if len(makes) == 1 else m) for m in f["models"])
        if f.get("variants") and len(f["models"]) == 1:
            names += " " + " / ".join(f["variants"])
        bits.append(names)
    elif makes:
        bits.append(" or ".join(makes))
    noun = f.get("body") or "vehicle"
    bits.append(noun if n == 1 else noun + "s")
    out = " ".join(bits)
    if f.get("years"):                                  # after the noun, so "1 2024 Swift" never reads as "12024"
        out += " (" + "/".join(str(y) for y in f["years"]) + ")"
    elif f.get("min_year"):
        out += f" ({f['min_year']} or newer)"
    if f.get("min_price") and f.get("max_price"):
        out += f" between {render.lakh(f['min_price'])} and {render.lakh(f['max_price'])}"
    elif f.get("max_price"):
        out += f" under {render.lakh(f['max_price'])}"
    elif f.get("min_price"):
        out += f" above {render.lakh(f['min_price'])}"
    if f.get("max_km"):
        out += f" with under {f['max_km']:,} km"
    return out


_FIELD_LABEL = {
    "variant": "variants", "color": "colours", "fuel": "fuel options", "transmission": "transmissions",
    "body": "body styles", "year": "model years", "model": "models", "make": "makes",
}
_GROUP_LABEL = {"make": "make", "model": "model", "fuel": "fuel type", "transmission": "transmission",
                "body": "body style", "type": "condition", "year": "model year", "color": "colour"}
_METRIC_LABEL = {
    "count": "vehicles", "avg_price": "average price", "min_price": "lowest price",
    "max_price": "highest price", "avg_km": "average mileage",
}


def _scope(f: dict[str, Any], subject: str = "vehicles in stock") -> str:
    scoped = {k: v for k, v in f.items() if k not in ("sort", "limit", "stock_ids")}
    return describe(scoped) if scoped else subject


def _fmt_metric(metric: str, v: float) -> str:
    if metric in ("avg_price", "min_price", "max_price"):
        return lakh(v)
    if metric == "avg_km":
        return f"{v:,.0f} km"
    return f"{int(v)}"


def render_result(res: Q.InventoryResult) -> tuple[str, dict[str, Any]]:
    """-> (text, ctx). ctx['result_ids'] is set only when concrete vehicles were shown."""
    q = res.query
    f = q.filters
    op = q.operation

    if op == Q.DETAIL:
        if not res.rows:
            return f"I couldn't find stock {q.stock_id} in our inventory.", {}
        v = res.rows[0]
        km = f"{v['mileage_km']:,} km" if v["type"] == "used" else "0 km (new)"
        body = f"{v['body']}, {v['seats']} seats" if v.get("body") else "n/a"
        return (
            f"{v['year']} {v['make']} {v['model']} {v['variant']} [{v['stock_id']}] - {v['type']}\n"
            f"- Price: {lakh(v['price'])}\n- Fuel / transmission: {v['fuel']} / {v['transmission']}\n"
            f"- Colour: {v['color']}\n- Mileage: {km}\n- Body / seats: {body}"
        ), {"result_ids": [v["stock_id"]]}

    if op == Q.COMPARE:
        names = " vs ".join(c["model"] for c in res.compared)
        blocks = []
        for c in res.compared:
            price = lakh(c["min_price"]) if c["min_price"] == c["max_price"] else f"{lakh(c['min_price'])} - {lakh(c['max_price'])}"
            lines = [
                f"{c['make']} {c['model']} - {c['count']} in stock",
                f"  Price: {price}",
                f"  Fuel: {', '.join(c['fuels'])}",
                f"  Transmission: {', '.join(c['transmissions'])}",
                f"  Year: {', '.join(str(y) for y in c['years'])} ({' / '.join(c['types'])})",
                f"  Variants: {', '.join(c['variants'])}",
            ]
            if c["body"]:
                lines.append(f"  Body / seats: {c['body']}, {c['seats']}")
            if c["max_km"] is not None:
                km = f"{c['min_km']:,} km" if c["min_km"] == c["max_km"] else f"{c['min_km']:,} - {c['max_km']:,} km"
                lines.append(f"  Mileage (used): {km}")
            blocks.append("\n".join(lines))
        cheapest = min(res.compared, key=lambda c: c["min_price"])
        tail = f"\nLowest starting price: {cheapest['make']} {cheapest['model']} at {lakh(cheapest['min_price'])}."
        ids = [sid for c in res.compared for sid in c["stock_ids"]]
        return f"{names} (current stock):\n" + "\n".join(blocks) + tail, {"result_ids": ids}

    if op == Q.DISTINCT:
        label = _FIELD_LABEL.get(q.field or "", q.field or "values")
        if not res.values:
            return f"I couldn't find any {_scope(f)} in stock, so there are no {label} to list.", {}
        who = _scope(f, "our stock")
        listing = ", ".join(f"{v} ({n})" if n > 1 else v for v, n in res.values)
        return f"{who[0].upper() + who[1:]} - {label} in stock: {listing}.", {}

    if op == Q.RANGE:
        if not res.stats:
            return f"I couldn't find any {_scope(f)} in stock.", {}
        s = res.stats
        return (
            f"Prices for {_scope(f)} range from {lakh(s['min'])} to {lakh(s['max'])} ({res.total} in stock).\n"
            f"Lowest: {vline(s['cheapest'])[2:]}\nHighest: {vline(s['dearest'])[2:]}"
        ), {}

    if op == Q.AVERAGE:
        if not res.stats:
            return f"I couldn't find any {_scope(f)} in stock, so there is no average.", {}
        s = res.stats
        what = "mileage" if s["metric"] == "avg_km" else "price"
        fmt = (lambda x: f"{x:,.0f} km") if s["metric"] == "avg_km" else lakh
        return (
            f"The average {what} across our {res.total} {_scope(f)} in stock is {fmt(s['value'])} "
            f"(lowest {fmt(s['min'])}, highest {fmt(s['max'])})."
        ), {}

    if op == Q.GROUP_BY:
        metric = q.metric or "count"
        if not res.groups:
            return f"I couldn't find any {_scope(f)} in stock.", {}
        by = _GROUP_LABEL.get(q.field or "", q.field or "group")
        if q.pick:
            top = res.groups[0]
            names = " and ".join(g[0] for g in res.groups)
            if metric == "min_price":
                return f"The {by} with the cheapest vehicle is {names} ({lakh(top[1])}).", {}
            if metric == "max_price":
                return f"The {by} with the most expensive vehicle is {names} ({lakh(top[1])}).", {}
            extreme = "highest" if q.pick == "max" else "lowest"
            what = "number of vehicles" if metric == "count" else _METRIC_LABEL[metric]
            unit = " vehicles" if metric == "count" else ""
            return f"The {by} with the {extreme} {what} is {names} ({_fmt_metric(metric, top[1])}{unit}).", {}
        head = {
            "count": f"{_scope(f).capitalize()} by {by}",
            "avg_price": f"Average price by {by} ({_scope(f)})",
            "min_price": f"Lowest price by {by} ({_scope(f)})",
            "max_price": f"Highest price by {by} ({_scope(f)})",
            "avg_km": f"Average mileage by {by} ({_scope(f)})",
        }[metric]
        lines = []
        for label, val, n in res.groups:
            if metric == "count":
                lines.append(f"- {label}: {int(val)}")
            else:
                lines.append(f"- {label}: {_fmt_metric(metric, val)} ({n} vehicle{'s' if n != 1 else ''})")
        return f"{head}:\n" + "\n".join(lines), {}

    # LIST / COUNT
    if op == Q.COUNT:
        n = res.total
        return (f"We have {n} {describe(f, n)} in stock." if n else f"We don't have any {describe(f)} in stock right now."), {}
    limit = settings.inventory_result_limit
    if not res.total:
        return f"I couldn't find any {describe(f)} in stock. Try widening the budget or dropping a filter.", {}
    shown = res.rows[:limit]
    head = f"{'Yes - w' if f.get('models') else 'W'}e have {res.total} {describe(f, res.total)} in stock."
    more = f"\n…and {res.total - limit} more - see the inventory panel." if res.total > limit else ""
    return f"{head}\n" + "\n".join(vline(r) for r in shown) + more, {"result_ids": [r["stock_id"] for r in shown]}
