import pytest

from app import inventory
from app.config import BASE_DIR
from app.router.intents import BOOK, CANCEL, SELECT_VEHICLE, RouteContext
from app.router.rules import is_book, is_cancel, is_select_vehicle, route


@pytest.fixture(autouse=True)
def _catalog():
    inventory._catalog = inventory.Catalog(
        inventory._load_json(BASE_DIR / "data" / "inventory.json"),
        inventory._load_json(BASE_DIR / "data" / "vehicle_attributes.json"),
    )


BOUND = RouteContext(customer_bound=True, customer_id="10001")


@pytest.mark.parametrize("text", [
    "don't pick s-1005",
    "do not choose s-1005",
    "i would never choose s-1003",
    "should i take s-1010 or s-1011?",
    "s-1010 or s-1011, which should i take?",
    "i'd like details about s-1005",
])
def test_negated_or_deliberated_selection_is_not_a_selection(text):
    assert not is_select_vehicle(text)
    assert route(text, BOUND).intent != SELECT_VEHICLE


@pytest.mark.parametrize("text", ["i'll take s-1005", "let's go with s-1005", "can i pick s-1005?"])
def test_plain_selection_still_selects(text):
    assert is_select_vehicle(text)
    assert route(text, BOUND).intent == SELECT_VEHICLE


@pytest.mark.parametrize("text", [
    "do not cancel my appointment",
    "don't cancel my test drive",
    "i don't want to cancel my appointment",
    "should i cancel my test drive?",
])
def test_negated_cancel_is_not_a_cancel(text):
    assert not is_cancel(text)
    assert route(text, BOUND).intent != CANCEL


@pytest.mark.parametrize("text", ["cancel my test drive", "can you cancel my appointment?"])
def test_plain_cancel_still_cancels(text):
    assert is_cancel(text)
    assert route(text, BOUND).intent == CANCEL


@pytest.mark.parametrize("text", [
    "can you book a test drive for me tomorrow at 10am?",
    "could you schedule a callback for tomorrow at 11?",
    "if it's not too late, book me a test drive tomorrow at 11",
])
def test_polite_or_unrelated_negation_still_books(text):
    assert is_book(text)
    assert route(text, BOUND).intent == BOOK


@pytest.mark.parametrize("text", ["don't book a test drive", "i never asked you to schedule a callback"])
def test_negated_booking_is_not_a_booking(text):
    assert not is_book(text)
    assert route(text, BOUND).intent != BOOK


# ---- multi-clause: never answer one clause and silently drop another --------------------

from app.router.intents import FIELD_LOOKUP, INVENTORY, LLM  # noqa: E402


@pytest.mark.parametrize("text", [
    "what's my email and does the creta have a sunroof?",
    "what is my quotation total and also which colour is best for me",
])
def test_partially_matched_message_goes_to_llm(text):
    r = route(text, BOUND)
    assert r.intent == LLM and r.hints.get("unmatched")


@pytest.mark.parametrize("text,intent", [
    ("book a test drive tomorrow at 10am and thanks", BOOK),
    ("book a test drive tomorrow at 10am, my wife will come too", BOOK),
    ("what is my name and my email", FIELD_LOOKUP),
])
def test_pleasantries_and_narrative_tails_do_not_force_llm(text, intent):
    assert route(text, BOUND).intent == intent


def test_numeric_range_is_not_split_into_clauses():
    r = route("give me all cars between 25 and 35 lakhs", BOUND)
    assert r.intent == INVENTORY
    assert r.slots["filters"]["min_price"] == 2_500_000 and r.slots["filters"]["max_price"] == 3_500_000


# ---- pathological input must not block the event loop ------------------------------------

def test_long_digit_runs_route_quickly():
    import time
    from app.router.extract import find_email, parse_price
    assert parse_price("between 25 and 35 lakhs") == (2_500_000, 3_500_000)
    assert parse_price("1.5-2 cr") == (15_000_000, 20_000_000)
    assert find_email("my email is (x@y") == (None, "(x@y")
    t = time.perf_counter()
    for n in (500, 1000, 2000):
        route("1" * n, BOUND)
    assert time.perf_counter() - t < 0.5
