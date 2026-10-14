import pytest

from app import inventory
from app.config import BASE_DIR
from app.prompting import focus_rows, prompt_inputs


@pytest.fixture(autouse=True)
def _catalog():
    inventory._catalog = inventory.Catalog(
        inventory._load_json(BASE_DIR / "data" / "inventory.json"),
        inventory._load_json(BASE_DIR / "data" / "vehicle_attributes.json"),
    )


def _inputs(question, session=None, record=None):
    return prompt_inputs(question=question, cid=None, record=record, appts=[], turns=[], session=session)


def test_followup_price_question_is_grounded_on_browsed_model():
    session = {"inv_filters": {"makes": ["Hyundai"], "models": ["Creta"]}}
    stock = _inputs("and its price?", session)["volatile"]
    assert "S-1010" in stock and "S-1011" in stock
    assert "Swift" not in stock                      # not the cheapest-10 catalogue dump


def test_followup_uses_rows_shown_last_turn():
    session = {"inv_filters": {"makes": ["Hyundai"]}, "inv_result_ids": ["S-1011"]}
    rows = focus_rows(session, None)
    assert [r["stock_id"] for r in rows] == ["S-1011"]


def test_selected_vehicle_is_in_focus_even_without_browsing_filters():
    record = {"enquiry": {"interested_stock_id": "S-1010"}}
    stock = _inputs("how many km has it done?", {}, record)["volatile"]
    assert "S-1010" in stock


def test_question_naming_its_own_model_ignores_focus():
    session = {"inv_filters": {"models": ["Creta"]}}
    stock = _inputs("price of the swift?", session)["volatile"]
    assert "Swift" in stock and "S-1010" not in stock
