from app import inventory
from app.config import BASE_DIR
from app.router.extract import parse_price
from app.router.rules import is_inventory
from app.router.intents import LLM, RouteContext
from app.router.rules import route


def test_unparseable_price_comparator_falls_through_to_llm():
    assert not is_inventory("i want to know number of vehicle that have price greater then 10 lacs")


def test_valid_price_comparator_stays_deterministic():
    assert is_inventory("let me know all vehicle price above 10 lacs")


def test_bare_price_does_not_create_an_approximate_range():
    assert parse_price("vehicles around 10 lacs") == (None, None)


def test_malformed_price_request_does_not_reuse_previous_filters():
    context = RouteContext(last_filters={"makes": ["Tata"], "min_price": 900000, "max_price": 1100000})

    result = route("i want to know number of vehicle that have price greater then 10 lacs", context)

    assert result.intent == LLM


def test_new_make_does_not_inherit_previous_price_filter():
    inventory._catalog = inventory.Catalog(
        inventory._load_json(BASE_DIR / "data" / "inventory.json"),
        inventory._load_json(BASE_DIR / "data" / "vehicle_attributes.json"),
    )
    context = RouteContext(last_filters={"max_price": 1000000})

    result = route("let me know all tata vehicles", context)

    assert result.slots["filters"] == {"makes": ["Tata"]}


def test_new_price_replaces_previous_price_bounds():
    context = RouteContext(last_filters={"min_price": 500000, "max_price": 1000000})

    result = route("vehicles above 50 lacs", context)

    assert result.slots["filters"]["min_price"] == 5_000_000
    assert "max_price" not in result.slots["filters"]