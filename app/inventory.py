"""Indexed in-process vehicle catalogue.

The old implementation re-read (and JSON-decoded) the *entire* stock list from Redis
twice per chat turn and pasted it into every LLM prompt. The catalogue is small,
read-mostly reference data, so it now lives in process memory with per-field indexes:

    make / model / type / fuel / transmission / body  ->  set of row positions

A search intersects the relevant index sets (smallest first), applies numeric
predicates (price, mileage, seats) to the survivors and returns only the top N rows.
Redis is used once at startup (to pick up externally-edited stock) and never on the
request path.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Iterable

from . import cache
from .config import BASE_DIR

log = logging.getLogger("inventory")

_INVENTORY_FILE = BASE_DIR / "data" / "inventory.json"
_ATTRS_FILE = BASE_DIR / "data" / "vehicle_attributes.json"

MAKE_ALIASES = {
    "maruti": "Maruti Suzuki", "suzuki": "Maruti Suzuki", "maruti suzuki": "Maruti Suzuki",
    "mg motor": "MG", "morris garages": "MG",
    "beemer": "BMW", "bmw": "BMW",
}


BASE_COLORS = {"white", "black", "red", "blue", "grey", "silver", "brown", "khaki", "olive"}
COLOR_ALIASES = {"gray": "grey"}


def _low(x: Any) -> str:
    return str(x).strip().lower()


class Catalog:
    def __init__(self, items: list[dict[str, Any]], attrs: dict[str, Any] | None = None) -> None:
        attrs = attrs or {}
        self.items: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for it in items:
            sid = it.get("stock_id")
            if sid is not None:
                if sid in seen_ids:
                    continue                                   # dedupe: a corrupted/double-seeded
                seen_ids.add(sid)                               # source must never surface twice.
            row = dict(it)
            body, seats = (attrs.get(it["make"], {}).get(it["model"]) or [None, None])
            row["body"] = body
            row["seats"] = seats
            self.items.append(row)

        self._idx: dict[str, dict[str, set[int]]] = {
            k: {} for k in ("make", "model", "type", "fuel", "transmission", "body", "year", "variant")
        }
        self._id_pos: dict[str, int] = {}
        for pos, row in enumerate(self.items):
            for field in self._idx:
                v = row.get(field)
                if v:
                    self._idx[field].setdefault(_low(v), set()).add(pos)
            sid = row.get("stock_id")
            if sid:
                self._id_pos[sid] = pos

        self.makes: list[str] = sorted({r["make"] for r in self.items})
        self.models_by_make: dict[str, list[str]] = {}
        for r in self.items:
            self.models_by_make.setdefault(r["make"], [])
            if r["model"] not in self.models_by_make[r["make"]]:
                self.models_by_make[r["make"]].append(r["model"])
        for m in self.models_by_make.values():
            m.sort()

        # (lowercase surface form, make, model) longest-first for vocabulary matching
        vocab = [(_low(model), make, model) for make, ms in self.models_by_make.items() for model in ms]
        vocab.sort(key=lambda t: -len(t[0]))
        self.model_vocab = vocab
        self.make_vocab = sorted(
            [(_low(m), m) for m in self.makes] + [(a, c) for a, c in MAKE_ALIASES.items() if c in self.makes],
            key=lambda t: -len(t[0]),
        )
        self._model_res = [(re.compile(rf"(?<![\w]){re.escape(s)}(?![\w])"), mk, mo) for s, mk, mo in vocab]
        self._make_res = [(re.compile(rf"(?<![\w]){re.escape(s)}s?(?![\w])"), c) for s, c in self.make_vocab]

        # (variant text, make, model, variant) longest-first. A variant is only ever matched
        # *together with* its model (see find_variants) because words like "Delta" or "Sport Line"
        # are not meaningful on their own.
        vv = {(r["make"], r["model"], r["variant"]) for r in self.items if r.get("variant")}
        self.variant_vocab = sorted(((_low(v), mk, mo, v) for mk, mo, v in vv), key=lambda t: -len(t[0]))
        self._variant_res = [
            (re.compile(rf"(?<![\w]){re.escape(t)}(?![\w])"), mk, mo, v) for t, mk, mo, v in self.variant_vocab
        ]
        # single colour words actually present in stock ("white" -> Pristine White, Mineral White ...)
        words = {w for r in self.items for w in re.findall(r"[a-z]+", _low(r.get("color", "")))}
        self.color_words: set[str] = {w for w in BASE_COLORS if w in words}
        self.years: list[int] = sorted({r["year"] for r in self.items if r.get("year")})

    # -- vocabulary matching (used by the router) ---------------------------------
    def find_models(self, text_lower: str) -> list[tuple[str, str]]:
        """Longest-match-first, non-overlapping model mentions -> [(make, model)]."""
        found: list[tuple[int, str, str]] = []
        taken: list[tuple[int, int]] = []
        for rx, make, model in self._model_res:
            for m in rx.finditer(text_lower):
                if any(not (m.end() <= a or m.start() >= b) for a, b in taken):
                    continue
                taken.append((m.start(), m.end()))
                found.append((m.start(), make, model))
        found.sort()
        return [(mk, mo) for _, mk, mo in found]

    def find_makes(self, text_lower: str) -> list[str]:
        found: list[tuple[int, str]] = []
        taken: list[tuple[int, int]] = []
        for rx, canon in self._make_res:
            for m in rx.finditer(text_lower):
                if any(not (m.end() <= a or m.start() >= b) for a, b in taken):
                    continue
                taken.append((m.start(), m.end()))
                found.append((m.start(), canon))
        found.sort()
        out: list[str] = []
        for _, c in found:
            if c not in out:
                out.append(c)
        return out

    def find_variants(self, text_lower: str, models: Iterable[str] | None = None) -> list[tuple[str, str, str]]:
        """Variant mentions -> [(make, model, variant)], longest-first and non-overlapping.

        When `models` is given only variants of those models are considered (the common case:
        "Swift ZXi AMT"); this avoids false positives on generic words such as "delta" or "sport"."""
        allowed = {_low(m) for m in models} if models is not None else None
        found: list[tuple[int, str, str, str]] = []
        taken: list[tuple[int, int]] = []
        for rx, make, model, variant in self._variant_res:
            if allowed is not None and _low(model) not in allowed:
                continue
            for m in rx.finditer(text_lower):
                if any(not (m.end() <= a or m.start() >= b) for a, b in taken):
                    continue
                taken.append((m.start(), m.end()))
                found.append((m.start(), make, model, variant))
        found.sort()
        return [(mk, mo, v) for _, mk, mo, v in found]

    def find_colors(self, text_lower: str) -> list[str]:
        """Base colour words that exist in stock and are mentioned in the text (gray -> grey)."""
        out: list[str] = []
        for w in re.findall(r"[a-z]+", text_lower):
            w = COLOR_ALIASES.get(w, w)
            if w in self.color_words and w not in out:
                out.append(w)
        return out

    def get(self, stock_id: str) -> dict[str, Any] | None:
        pos = self._id_pos.get(stock_id) or self._id_pos.get((stock_id or "").upper())
        return self.items[pos] if pos is not None else None

    # -- queries -------------------------------------------------------------------
    def search(
        self,
        *,
        makes: Iterable[str] | None = None,
        models: Iterable[str] | None = None,
        vtype: str | None = None,
        fuel: str | None = None,
        transmission: str | None = None,
        body: str | None = None,
        min_price: int | None = None,
        max_price: int | None = None,
        max_km: int | None = None,
        min_seats: int | None = None,
        max_seats: int | None = None,
        years: Iterable[int] | None = None,       # exact model years
        min_year: int | None = None,
        colors: Iterable[str] | None = None,      # colour words, matched as whole words in the colour name
        variants: Iterable[str] | None = None,    # exact variant names
        sort: str | None = None,          # "price_asc" | "price_desc" | "km_asc"
        limit: int | None = None,
        stock_ids: Iterable[str] | None = None,   # restrict to a specific previously-shown set ("these")
    ) -> tuple[int, list[dict[str, Any]]]:
        """Return (total_matches, first `limit` rows)."""
        sets: list[set[int]] = []

        if stock_ids is not None:
            sets.append({self._id_pos[sid] for sid in stock_ids if sid in self._id_pos})

        def add(field: str, values: Iterable[str] | str | None) -> bool:
            if not values:
                return True
            vals = [values] if isinstance(values, str) else list(values)
            union: set[int] = set()
            for v in vals:
                union |= self._idx[field].get(_low(v), set())
            sets.append(union)
            return bool(union)

        ok = True
        ok &= add("make", makes)
        ok &= add("model", models)
        if vtype and vtype != "all":
            ok &= add("type", vtype)
        ok &= add("fuel", fuel)
        ok &= add("transmission", transmission)
        ok &= add("body", body)
        ok &= add("year", [str(y) for y in years] if years else None)
        ok &= add("variant", variants)
        if not ok:
            return 0, []

        if sets:
            sets.sort(key=len)
            cand = set.intersection(*sets)
        else:
            cand = set(range(len(self.items)))

        rows = [self.items[i] for i in cand]
        if min_price is not None:
            rows = [r for r in rows if r["price"] >= min_price]
        if max_price is not None:
            rows = [r for r in rows if r["price"] <= max_price]
        if max_km is not None:
            rows = [r for r in rows if r["mileage_km"] <= max_km]
        if min_seats is not None:
            rows = [r for r in rows if (r.get("seats") or 0) >= min_seats]
        if max_seats is not None:
            rows = [r for r in rows if 0 < (r.get("seats") or 0) <= max_seats]
        if min_year is not None:
            rows = [r for r in rows if (r.get("year") or 0) >= min_year]
        if colors:
            wanted = {COLOR_ALIASES.get(_low(c), _low(c)) for c in ([colors] if isinstance(colors, str) else colors)}
            rows = [r for r in rows if wanted & set(re.findall(r"[a-z]+", _low(r.get("color", ""))))]

        key = {
            "price_asc": lambda r: (r["price"], r["stock_id"]),
            "price_desc": lambda r: (-r["price"], r["stock_id"]),
            "km_asc": lambda r: (r["mileage_km"], r["stock_id"]),
            "year_desc": lambda r: (-(r.get("year") or 0), r["stock_id"]),
            "year_asc": lambda r: (r.get("year") or 0, r["stock_id"]),
        }.get(sort or "", lambda r: r["stock_id"])
        rows.sort(key=key)
        total = len(rows)
        return total, rows[:limit] if limit else rows

    def counts(self) -> dict[str, int]:
        return {
            "total": len(self.items),
            "new_cars": len(self._idx["type"].get("new", ())),
            "used_cars": len(self._idx["type"].get("used", ())),
            "makes": len(self.makes),
        }


_catalog: Catalog = Catalog([])


def catalog() -> Catalog:
    return _catalog


def _load_json(path) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not load %s: %s", path.name, exc)
        return None


async def init() -> int:
    """Build the in-memory catalogue once at startup. Returns vehicle count."""
    global _catalog
    items = await cache.load_inventory()                 # Redis copy, if someone edited stock there
    if not items:
        items = _load_json(_INVENTORY_FILE) or []
        if items:
            await cache.seed_inventory(items)
    attrs = _load_json(_ATTRS_FILE) or {}
    _catalog = Catalog(items, attrs)
    log.info("inventory ready: %d vehicles, %d makes", len(_catalog.items), len(_catalog.makes))
    return len(_catalog.items)


# ---- thin helpers used by REST routes ---------------------------------------------

def list_makes() -> list[str]:
    return list(_catalog.makes)


def list_models(make: str) -> list[str]:
    for m, models in _catalog.models_by_make.items():
        if m.lower() == make.lower():
            return list(models)
    return []


def count_inventory() -> dict[str, int]:
    return _catalog.counts()


def search(**kw: Any) -> list[dict[str, Any]]:
    return _catalog.search(**kw)[1]