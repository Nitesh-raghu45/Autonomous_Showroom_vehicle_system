"""Inventory domain: catalogue indexes, structured query engine, router and conversation flow.

Everything here is deterministic - no LLM is involved for any inventory question the rules can parse.
"""
import json

import httpx
import pytest
import pytest_asyncio

from app import actions, inventory, inventory_query as Q
from app.config import BASE_DIR
from app.router import extract as X
from app.router import rules
from app.router.intents import INVENTORY, LLM, RouteContext

ITEMS = json.load(open(BASE_DIR / "data" / "inventory.json", encoding="utf-8"))
ATTRS = json.load(open(BASE_DIR / "data" / "vehicle_attributes.json", encoding="utf-8"))


@pytest.fixture(scope="module", autouse=True)
def catalog():
    prev = inventory._catalog
    inventory._catalog = inventory.Catalog(ITEMS, ATTRS)
    yield inventory._catalog
    inventory._catalog = prev


def run(**kw):
    return Q.execute(Q.InventoryQuery(**kw))


# ---------------------------------------------------------------------------
# catalogue: new indexes / predicates
# ---------------------------------------------------------------------------

def test_year_variant_color_filters(catalog):
    assert catalog.search(variants=["ZXi AMT"])[0] == 1
    total, rows = catalog.search(years=[2022])
    assert total == 2 and all(r["year"] == 2022 for r in rows)
    assert catalog.search(min_year=2023)[0] == sum(1 for i in ITEMS if i["year"] >= 2023)
    white = catalog.search(colors=["white"])[1]
    assert white and all("white" in r["color"].lower() for r in white)
    assert catalog.search(colors=["gray"])[0] == catalog.search(colors=["grey"])[0] > 0   # alias


def test_exact_seat_count(catalog):
    _, rows = catalog.search(min_seats=5, max_seats=5)
    assert rows and {r["seats"] for r in rows} == {5}


def test_get_by_stock_id(catalog):
    assert catalog.get("S-1001")["model"] == "Swift"
    assert catalog.get("S-9999") is None


def test_variants_only_match_with_their_model(catalog):
    assert catalog.find_variants("zxi amt", ["Swift"]) == [("Maruti Suzuki", "Swift", "ZXi AMT")]
    assert catalog.find_variants("delta", ["Swift"]) == []          # Delta belongs to Baleno


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("question", "expected"), [
    ("do you have a 2024 swift zxi amt", {"models": ["Swift"], "variants": ["ZXi AMT"], "years": [2024]}),
    ("5 cheapest cars", {"sort": "price_asc", "limit": 5}),
    ("white diesel suvs after 2022", {"colors": ["white"], "fuel": "Diesel", "body": "suv", "min_year": 2023}),
    ("5 seater sedans", {"min_seats": 5, "max_seats": 5, "body": "sedan"}),
    ("show me 7 seater suvs under 30 lakhs", {"min_seats": 7, "body": "suv", "max_price": 3_000_000}),
    ("newest swift", {"sort": "year_desc", "models": ["Swift"]}),
])
def test_extract_inventory_filters(question, expected):
    got = X.inventory_filters(question)
    for k, v in expected.items():
        assert got.get(k) == v, (question, k, got)


@pytest.mark.parametrize("question", ["book a test drive on 2026-10-10", "cheapest 5 lakh car", "under 2000 km"])
def test_year_is_not_confused_with_dates_prices_or_km(question):
    assert "years" not in X.inventory_filters(question) and "limit" not in X.inventory_filters(question)


# ---------------------------------------------------------------------------
# engine: exact numbers, computed from the catalogue
# ---------------------------------------------------------------------------

def test_count_and_list_match_catalogue():
    assert run(operation="count", filters={"body": "suv"}).total == sum(
        1 for i in ITEMS if (ATTRS.get(i["make"], {}).get(i["model"]) or [None])[0] == "suv")
    assert len(run(operation="list", filters={"sort": "price_asc", "limit": 5}).rows) == 5


def test_group_by_make_sums_to_total():
    res = run(operation="group_by", field="make", metric="count")
    assert sum(n for _, _, n in res.groups) == len(ITEMS)
    assert {m for m, _, _ in res.groups} == {i["make"] for i in ITEMS}


def test_group_by_pick_returns_the_winner():
    res = run(operation="group_by", field="make", metric="avg_price", pick="max")
    by_make = {}
    for i in ITEMS:
        by_make.setdefault(i["make"], []).append(i["price"])
    best = max(by_make, key=lambda m: sum(by_make[m]) / len(by_make[m]))
    assert [g[0] for g in res.groups] == [best]


def test_average_and_range():
    tata = [i["price"] for i in ITEMS if i["make"] == "Tata"]
    avg = run(operation="average", filters={"makes": ["Tata"]}).stats
    assert avg["value"] == pytest.approx(sum(tata) / len(tata)) and avg["min"] == min(tata) and avg["max"] == max(tata)
    rng = run(operation="range", filters={"makes": ["BMW"]}).stats
    assert rng["cheapest"]["price"] == rng["min"] <= rng["max"]


def test_distinct_values():
    res = run(operation="distinct", field="variant", filters={"models": ["Creta"]})
    assert [v for v, _ in res.values] == ["SX Diesel", "SX(O) Diesel"]
    fuels = run(operation="distinct", field="fuel", filters={"makes": ["BMW"]}).values
    assert sum(n for _, n in fuels) == sum(1 for i in ITEMS if i["make"] == "BMW")


def test_compare_and_detail():
    res = run(operation="compare", compare=[("Maruti Suzuki", "Swift"), ("Maruti Suzuki", "Baleno")])
    assert [c["model"] for c in res.compared] == ["Swift", "Baleno"] and all(c["count"] == 2 for c in res.compared)
    assert run(operation="detail", stock_id="S-1001").rows[0]["variant"] == "ZXi AMT"
    assert run(operation="detail", stock_id="S-0000").rows == []


@pytest.mark.parametrize("bad", [
    dict(operation="drop_table"), dict(operation="group_by", field="price"),
    dict(operation="distinct", field="stock_id"), dict(operation="average", metric="median"),
])
def test_invalid_queries_are_rejected(bad):
    with pytest.raises(ValueError):
        run(**bad)


def test_invalid_structured_query_gives_a_safe_answer():
    r = actions.search_inventory({}, "group", {"field": "price"})
    assert "rephrase" in r.text.lower()


# ---------------------------------------------------------------------------
# router: question -> structured query, no LLM
# ---------------------------------------------------------------------------

CTX = RouteContext(customer_bound=True, customer_id="10001", customer_name="Test User")


@pytest.mark.parametrize(("question", "mode", "extra"), [
    ("How many cars do we have from each brand?", "group", {"field": "make", "metric": "count"}),
    ("How many cars are there from each fuel type?", "group", {"field": "fuel", "metric": "count"}),
    ("Which brand has the highest average price?", "group", {"field": "make", "metric": "avg_price", "pick": "max"}),
    ("which make has the most cars", "group", {"field": "make", "metric": "count", "pick": "max"}),
    ("which fuel type has the fewest vehicles", "group", {"field": "fuel", "pick": "min"}),
    ("compare average price of new vs used cars", "group", {"field": "type", "metric": "avg_price"}),
    ("What is the average price of Tata cars?", "average", {"metric": "avg_price"}),
    ("what is the price range of BMW", "range", {}),
    ("What variants of Creta do you have?", "distinct", {"field": "variant"}),
    ("What fuel options do you have for BMW?", "distinct", {"field": "fuel"}),
    ("tell me about S-1005", "detail", {"stock_id": "S-1005"}),
    ("Compare Swift and Baleno", "compare", {}),
    ("Creta vs Seltos", "compare", {}),
    ("Give me the 5 cheapest cars.", "superlative", {}),
    ("newest swift", "superlative", {}),
    ("How many Tata cars are available?", "count", {}),
])
def test_router_builds_structured_query(question, mode, extra):
    r = rules.route(question, CTX)
    assert r.intent == INVENTORY and r.slots["mode"] == mode, (question, r)
    for k, v in extra.items():
        assert r.slots.get(k) == v, (question, k, r.slots)


@pytest.mark.parametrize("question", [
    "Compare a new Kia Seltos and a used BMW X1 - which is cheaper?",    # verdict + new/used -> LLM
    "Which is better, Creta or Seltos?",                                   # judgement call -> LLM
])
def test_judgement_questions_still_go_to_the_llm(question):
    assert rules.route(question, CTX).intent == LLM


def test_record_questions_are_not_hijacked_by_aggregates():
    assert rules.route("which make am I interested in", CTX).intent != INVENTORY


def test_analytics_do_not_overwrite_browsing_context():
    r = rules.route("how many cars per brand", CTX)
    assert r.slots["sticky"] is False


# ---------------------------------------------------------------------------
# rendered answers (spot checks against the data)
# ---------------------------------------------------------------------------

def text(question, ctx=CTX):
    r = rules.route(question, ctx)
    return actions.search_inventory(r.slots["filters"], r.slots["mode"], r.slots).text


def test_rendered_answers():
    assert "8" in text("how many maruti cars do you have") and "Maruti Suzuki" in text("how many maruti cars do you have")
    out = text("How many cars do we have from each brand?")
    assert out.startswith("Vehicles in stock by make:") and "- BMW: 7" in out
    assert "BMW" in text("Which brand has the highest average price?")
    assert "Swift vs Baleno" in text("compare swift and baleno") and "Variants:" in text("compare swift and baleno")
    assert "[S-1001]" in text("do you have a 2024 swift zxi amt") and text("do you have a 2024 swift zxi amt").count("[S-") == 1
    assert "couldn't find stock S-9999" in text("details of S-9999")
    assert "couldn't find any" in text("do you have a 2025 swift").lower()


# ---------------------------------------------------------------------------
# LLM planner slots -> validated structured query
# ---------------------------------------------------------------------------

def test_planner_slots_are_validated_and_mapped():
    f, mode, extra = Q.slots_to_query({"make": "tata", "body": "suv", "transmission": "automatic", "max_price": "2500000",
                                       "min_seats": 7, "sort": "price_asc", "limit": 3})
    assert f == {"makes": ["Tata"], "body": "suv", "transmission": "Automatic", "max_price": 2_500_000,
                 "min_seats": 7, "sort": "price_asc", "limit": 3}
    assert mode == "superlative"

    f, mode, extra = Q.slots_to_query({"operation": "group_by", "field": "make", "metric": "avg_price"})
    assert (mode, extra["field"], extra["metric"], extra["sticky"]) == ("group", "make", "avg_price", False)

    # junk from the model is dropped / downgraded, never executed
    f, mode, extra = Q.slots_to_query({"operation": "group_by", "field": "price", "body": "spaceship", "max_price": "cheap"})
    assert mode == "list" and f == {} and "field" not in extra


# ---------------------------------------------------------------------------
# multi-turn conversation through the real app (fakeredis, mock LLM)
# ---------------------------------------------------------------------------

pytestmark_flow = pytest.mark.asyncio(loop_scope="module")


@pytest_asyncio.fixture(scope="module")
async def client():
    from app.llm import http as llm_http
    from app.main import app
    from scripts import mock_llm_server

    llm_http._client = httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_llm_server.app), base_url="http://mock-llm")
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
            yield c


async def chat(client, q, sid=None):
    r = await client.post("/chat", json={"session_id": sid, "customer_number": "10001", "question": q})
    assert r.status_code == 200, r.text
    return r.json()


@pytest.mark.asyncio(loop_scope="module")
async def test_conversation_refines_then_ranks_the_shown_set(client):
    d1 = await chat(client, "Show me Tata SUVs under 25 lakhs")
    sid = d1["session_id"]
    assert d1["stats"]["route"] == "router" and "Tata" in d1["answer"]

    d2 = await chat(client, "Which ones are automatic?", sid)
    assert d2["stats"]["route"] == "router" and "automatic" in d2["answer"].lower() and "Tata" in d2["answer"]

    d3 = await chat(client, "Which one is the cheapest among these?", sid)
    assert d3["stats"]["route"] == "router" and d3["answer"].count("[S-") == 1 and "Tata" in d3["answer"]


@pytest.mark.asyncio(loop_scope="module")
async def test_analytics_in_the_middle_do_not_reset_the_context(client):
    d1 = await chat(client, "Show me Tata SUVs")
    sid = d1["session_id"]
    d2 = await chat(client, "How many cars do we have from each brand?", sid)
    assert d2["stats"]["route"] == "router" and d2["stats"]["llm_ms"] == 0 and "- Tata: 7" in d2["answer"]
    d3 = await chat(client, "only automatic ones", sid)
    assert "Tata" in d3["answer"] and "automatic" in d3["answer"].lower()


@pytest.mark.asyncio(loop_scope="module")
async def test_new_inventory_operations_use_no_llm(client):
    for q in ("Compare Swift and Baleno", "What variants of Creta do you have?", "What is the average price of Tata cars?",
              "tell me about S-1005"):
        d = await chat(client, q)
        assert d["stats"]["route"] == "router" and d["stats"]["llm_ms"] == 0, (q, d["stats"])
        assert d["stats"]["usage"]["completion_tokens"] == 0 and d["stats"]["latency_ms"] < 500
