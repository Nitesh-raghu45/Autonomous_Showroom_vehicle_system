"""Inventory query engine: structured query -> in-memory catalogue -> exact result.

The architecture the inventory domain follows::

    question -> router/parser -> InventoryQuery -> execute() -> InventoryResult -> renderer

Nothing downstream of the parser ever sees English. A query is a small, explicit data
structure (operation + filters + optional field/metric), so the SAME executor serves

  * the deterministic rule router   (regex/vocabulary extraction -> InventoryQuery)
  * the LLM planner                 (model emits slots -> InventoryQuery)
  * the REST layer / tests          (build an InventoryQuery directly)

and every number in an answer (counts, min/max/average, group totals) is computed from the
indexed catalogue, never produced by a model. No Redis, no embeddings, no vector search:
the catalogue is structured transactional data, so indexed filtering is both faster and exact.

Operations
----------
    LIST      matching vehicles (default)           COUNT     how many match
    DISTINCT  distinct values of one field          RANGE     min / max price
    AVERAGE   mean price or mileage                 GROUP_BY  count / avg / min / max per make, fuel ...
    COMPARE   side-by-side models                   DETAIL    one vehicle by stock id
(superlatives - "cheapest", "5 most expensive" - are LIST with sort + limit.)
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as _field
from statistics import mean
from typing import Any

from . import inventory

# ---- operations ------------------------------------------------------------------------
LIST, COUNT, DISTINCT, RANGE, AVERAGE, GROUP_BY, COMPARE, DETAIL = (
    "list", "count", "distinct", "range", "average", "group_by", "compare", "detail",
)
OPERATIONS = (LIST, COUNT, DISTINCT, RANGE, AVERAGE, GROUP_BY, COMPARE, DETAIL)

# ---- fields ----------------------------------------------------------------------------
GROUP_FIELDS = ("make", "model", "fuel", "transmission", "body", "type", "year", "color")
DISTINCT_FIELDS = ("variant", "color", "fuel", "transmission", "body", "year", "model", "make")
METRICS = ("count", "avg_price", "min_price", "max_price", "avg_km")

# filter keys understood by Catalog.search (and therefore by every InventoryQuery)
_FILTER_KEYS = (
    "makes", "models", "vtype", "fuel", "fuels", "transmission", "body", "min_price", "max_price",
    "max_km", "min_seats", "max_seats", "years", "min_year", "colors", "variants", "sort", "limit",
    "stock_ids",
)


def search_kwargs(f: dict[str, Any]) -> dict[str, Any]:
    """Translate a filter dict into Catalog.search() keyword arguments."""
    return dict(
        makes=f.get("makes"), models=f.get("models"), vtype=f.get("vtype"),
        fuel=f.get("fuel") or f.get("fuels"), transmission=f.get("transmission"), body=f.get("body"),
        min_price=f.get("min_price"), max_price=f.get("max_price"), max_km=f.get("max_km"),
        min_seats=f.get("min_seats"), max_seats=f.get("max_seats"), years=f.get("years"),
        min_year=f.get("min_year"), colors=f.get("colors"), variants=f.get("variants"),
        sort=f.get("sort"), stock_ids=f.get("stock_ids"),
    )


@dataclass
class InventoryQuery:
    operation: str = LIST
    filters: dict[str, Any] = _field(default_factory=dict)
    field: str | None = None                    # DISTINCT / GROUP_BY field
    metric: str | None = None                   # GROUP_BY / AVERAGE metric
    pick: str | None = None                     # GROUP_BY: "max" | "min" -> only the winning group(s)
    compare: list[tuple[str, str]] = _field(default_factory=list)   # COMPARE: [(make, model), ...]
    stock_id: str | None = None                 # DETAIL

    # -- construction -----------------------------------------------------------------------
    @classmethod
    def from_slots(cls, slots: dict[str, Any], mode: str | None = None) -> "InventoryQuery":
        """Build a query from a router Route's slots ({filters, mode, field, metric, ...})."""
        mode = mode or slots.get("mode") or LIST
        op = {
            "list": LIST, "superlative": LIST, "models": LIST, "colors": DISTINCT, "makes": DISTINCT,
            "count": COUNT, "distinct": DISTINCT, "range": RANGE, "average": AVERAGE, "group": GROUP_BY,
            "compare": COMPARE, "detail": DETAIL,
        }.get(mode, LIST)
        fld = slots.get("field") or {"colors": "color", "makes": "make"}.get(mode)
        return cls(
            operation=op, filters=dict(slots.get("filters") or {}), field=fld, metric=slots.get("metric"),
            pick=slots.get("pick"), compare=[tuple(x) for x in slots.get("compare") or []],
            stock_id=slots.get("stock_id"),
        )

    def validate(self) -> "InventoryQuery":
        if self.operation not in OPERATIONS:
            raise ValueError(f"unknown inventory operation {self.operation!r}")
        if self.operation == GROUP_BY and self.field not in GROUP_FIELDS:
            raise ValueError(f"cannot group by {self.field!r}")
        if self.operation == DISTINCT and self.field not in DISTINCT_FIELDS:
            raise ValueError(f"cannot list distinct {self.field!r}")
        if self.metric is not None and self.metric not in METRICS:
            raise ValueError(f"unknown metric {self.metric!r}")
        return self


@dataclass
class InventoryResult:
    query: InventoryQuery
    total: int = 0                                  # vehicles matching the filters
    rows: list[dict[str, Any]] = _field(default_factory=list)       # LIST / COUNT sample / DETAIL
    values: list[tuple[str, int]] = _field(default_factory=list)    # DISTINCT: (value, vehicles)
    groups: list[tuple[str, float, int]] = _field(default_factory=list)   # GROUP_BY: (label, metric, vehicles)
    stats: dict[str, Any] = _field(default_factory=dict)            # AVERAGE / RANGE
    compared: list[dict[str, Any]] = _field(default_factory=list)   # COMPARE: one summary per model


# ---------------------------------------------------------------------------------------
# executor
# ---------------------------------------------------------------------------------------

def execute(query: InventoryQuery, cat: inventory.Catalog | None = None) -> InventoryResult:
    """Run a structured query against the in-memory catalogue. Pure and synchronous."""
    cat = cat or inventory.catalog()
    q = query.validate()
    res = InventoryResult(q)

    if q.operation == DETAIL:
        row = cat.get(q.stock_id or "")
        res.rows = [row] if row else []
        res.total = len(res.rows)
        return res

    if q.operation == COMPARE:
        for make, model in q.compare:
            _, rows = cat.search(makes=[make], models=[model])
            res.compared.append(_summarise(make, model, rows))
        res.total = sum(c["count"] for c in res.compared)
        return res

    total, rows = cat.search(**search_kwargs(q.filters))
    res.total = total

    if q.operation in (LIST, COUNT):
        limit = q.filters.get("limit")
        res.rows = rows[:limit] if limit else rows
    elif q.operation == DISTINCT:
        res.values = _distinct(rows, q.field or "")
    elif q.operation == RANGE:
        if rows:
            lo, hi = min(rows, key=lambda r: r["price"]), max(rows, key=lambda r: r["price"])
            res.stats = {"min": lo["price"], "max": hi["price"], "cheapest": lo, "dearest": hi}
    elif q.operation == AVERAGE:
        metric = q.metric or "avg_price"
        if rows:
            col = "mileage_km" if metric == "avg_km" else "price"
            res.stats = {"metric": metric, "value": mean(r[col] for r in rows), "min": min(r[col] for r in rows),
                         "max": max(r[col] for r in rows)}
    elif q.operation == GROUP_BY:
        res.groups = _group(rows, q.field or "make", q.metric or "count", q.pick)
    return res


def _key(row: dict[str, Any], fld: str) -> str:
    v = row.get(fld)
    if fld == "make" or fld == "model":
        return str(v)
    if v in (None, ""):
        return "Unspecified"
    return str(v).title() if fld in ("type", "body") else str(v)


def _distinct(rows: list[dict[str, Any]], fld: str) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for r in rows:
        v = _key(r, fld)
        counts[v] = counts.get(v, 0) + 1
    return sorted(counts.items(), key=lambda kv: kv[0].lower() if fld != "year" else -int(kv[0]) if kv[0].isdigit() else 0)


def _group(rows: list[dict[str, Any]], fld: str, metric: str, pick: str | None) -> list[tuple[str, float, int]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        buckets.setdefault(_key(r, fld), []).append(r)

    def value(rs: list[dict[str, Any]]) -> float:
        if metric == "avg_price":
            return mean(r["price"] for r in rs)
        if metric == "min_price":
            return min(r["price"] for r in rs)
        if metric == "max_price":
            return max(r["price"] for r in rs)
        if metric == "avg_km":
            return mean(r["mileage_km"] for r in rs)
        return float(len(rs))

    out = [(label, value(rs), len(rs)) for label, rs in buckets.items()]
    ascending = metric == "min_price" and pick != "max"
    out.sort(key=lambda t: (t[1] if ascending else -t[1], t[0].lower()))
    if pick and out:
        best = (min if pick == "min" else max)(v for _, v, _ in out)
        out = [t for t in out if t[1] == best]
    return out


def _summarise(make: str, model: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    def uniq(k: str) -> list[Any]:
        return sorted({r[k] for r in rows if r.get(k) not in (None, "")}, key=str)

    prices = [r["price"] for r in rows]
    used = [r["mileage_km"] for r in rows if r["type"] == "used"]
    return {
        "make": make, "model": model, "count": len(rows),
        "min_price": min(prices) if prices else None, "max_price": max(prices) if prices else None,
        "fuels": uniq("fuel"), "transmissions": uniq("transmission"), "years": uniq("year"),
        "types": uniq("type"), "variants": uniq("variant"), "colors": uniq("color"),
        "body": (uniq("body") or [None])[0], "seats": (uniq("seats") or [None])[0],
        "min_km": min(used) if used else None, "max_km": max(used) if used else None,
        "stock_ids": [r["stock_id"] for r in rows],
    }


# ---------------------------------------------------------------------------------------
# LLM planner -> InventoryQuery
# ---------------------------------------------------------------------------------------
# For questions the rules cannot parse ("something premium but practical for a family of seven,
# automatic, not too expensive") the model's ONLY job is to describe the query as slots. It never
# answers from memory: the slots are validated/clamped here and executed by the same engine.

_PLANNER_OPS = {
    "search": "list", "list": "list", "count": "count", "distinct": "distinct", "average": "average",
    "range": "range", "group_by": "group", "group": "group",
}
_BODIES = {"suv", "sedan", "hatchback", "mpv"}
_SORTS = {"price_asc", "price_desc", "km_asc", "year_desc", "year_asc"}


def _int(v: Any) -> int | None:
    try:
        return int(float(str(v).replace(",", "")))
    except (TypeError, ValueError):
        return None


def slots_to_query(slots: dict[str, Any], is_placeholder=lambda v: False) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Planner slots -> (filters, mode, extra_slots) ready for actions.search_inventory()."""
    cat = inventory.catalog()
    f: dict[str, Any] = {}

    def good(k: str) -> Any:
        v = slots.get(k)
        return v if v not in (None, "") and not is_placeholder(v) else None

    if good("make"):
        found = cat.find_makes(str(slots["make"]).lower())
        f["makes"] = found[:1] or [str(slots["make"])]
    if good("model"):
        found = cat.find_models(str(slots["model"]).lower())
        f["models"] = [found[0][1]] if found else [str(slots["model"])]
        if found and "makes" not in f:
            f["makes"] = [found[0][0]]
    if good("variant") and f.get("models"):
        vs = cat.find_variants(str(slots["variant"]).lower(), f["models"])
        f["variants"] = [vs[0][2]] if vs else [str(slots["variant"])]
    if good("vtype") in ("new", "used"):
        f["vtype"] = slots["vtype"]
    if good("fuel"):
        f["fuel"] = str(slots["fuel"]).strip().title() if str(slots["fuel"]).lower() != "cng" else "CNG"
    if str(good("transmission") or "").lower() in ("automatic", "manual"):
        f["transmission"] = str(slots["transmission"]).title()
    if str(good("body") or "").lower() in _BODIES:
        f["body"] = str(slots["body"]).lower()
    for k in ("min_price", "max_price", "min_seats", "max_km", "min_year"):
        n = _int(good(k))
        if n is not None and n > 0:
            f[k] = n
    yr = _int(good("year"))
    if yr and 1990 <= yr <= 2035:
        f["years"] = [yr]
    if good("color"):
        cols = cat.find_colors(str(slots["color"]).lower())
        if cols:
            f["colors"] = cols
    if good("sort") in _SORTS:
        f["sort"] = slots["sort"]
    lim = _int(good("limit"))
    if lim and 1 < lim <= 20:
        f["limit"] = lim

    op = _PLANNER_OPS.get(str(good("operation") or "list").lower(), "list")
    extra: dict[str, Any] = {}
    fld, metric = str(good("field") or "").lower() or None, str(good("metric") or "").lower() or None
    if op == "distinct" and fld in DISTINCT_FIELDS:
        extra["field"] = fld
    elif op == "group" and fld in GROUP_FIELDS:
        extra["field"] = fld
        extra["metric"] = metric if metric in METRICS else "count"
        extra["sticky"] = False
    elif op in ("distinct", "group"):
        op = "list"                                  # the model asked for something we can't compute exactly
    if op == "average":
        extra["metric"] = "avg_km" if metric == "avg_km" else "avg_price"
        extra["sticky"] = False
    if op == "range":
        extra["sticky"] = False
    if op == "list" and f.get("sort"):
        op = "superlative"
    return f, op, extra
