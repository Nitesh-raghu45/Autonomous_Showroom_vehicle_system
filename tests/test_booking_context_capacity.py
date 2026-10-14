"""Bookings carry the vehicle under discussion, and slots have a showroom-wide capacity."""
import httpx
import pytest
import pytest_asyncio

from app.config import settings

pytestmark = pytest.mark.asyncio(loop_scope="module")


@pytest_asyncio.fixture(scope="module")
async def client():
    from app.llm import http as llm_http
    from app.main import app
    from scripts import mock_llm_server

    llm_http._client = httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_llm_server.app), base_url="http://mock-llm")
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
            yield c


async def chat(client, cid, q, sid=None):
    r = await client.post("/chat", json={"session_id": sid, "customer_number": cid, "question": q})
    assert r.status_code == 200, r.text
    return r.json()


async def test_booking_for_it_carries_the_selected_vehicle(client):
    sid = "s-book-it"
    d = await chat(client, "10003", "I'll take S-1010", sid)
    assert d["action"]["type"] == "SELECT_VEHICLE"
    d = await chat(client, "10003", "book a test drive for it on 29th September at 11am", sid)
    assert d["action"]["type"] == "BOOK_APPOINTMENT"
    assert "S-1010" in (d["action"]["result"]["notes"] or "")


async def test_test_drive_slot_has_showroom_wide_capacity(client):
    cap = settings.slot_capacity("TEST_DRIVE")
    customers = ["10005", "10006", "10007", "10008", "10009"][: cap + 1]
    results = [await chat(client, cid, "book a test drive on 30th September at 3pm", f"s-cap-{cid}") for cid in customers]
    booked = [r for r in results if r["action"] and r["action"]["type"] == "BOOK_APPOINTMENT"]
    assert len(booked) == cap
    assert "fully booked" in results[-1]["answer"]

    # cancelling frees the seat for the next customer
    first = booked[0]["action"]["result"]
    r = await client.patch(f"/customer/{first['customer_id']}/schedule/{first['appointment_id']}/cancel")
    assert r.status_code == 200
    again = await chat(client, customers[-1], "book a test drive on 30th September at 3pm", "s-cap-retry")
    assert again["action"] and again["action"]["type"] == "BOOK_APPOINTMENT"


async def test_variant_followup_and_misspelled_relative_date_booking(client):
    """Replays a real conversation: browse BMWs, ask about 'Sport line', then book with a
    misspelled 'day after tommorow' and two out-of-hours times before a valid one."""
    sid, cid = "s-sportline", "10010"
    d = await chat(client, cid, "show me BMW cars under 50 lakhs", sid)
    assert "S-1031" in d["answer"] and "S-1030" in d["answer"]

    d = await chat(client, cid, "give me more details about Sport line", sid)
    assert "S-1031" in d["answer"] and "S-1030" not in d["answer"]

    d = await chat(client, cid, "book a test drive", sid)
    assert d["action"] is None and "Which date" in d["answer"]   # no "which car?" - the Sport Line is in focus

    d = await chat(client, cid, "day after tommorow 9PM", sid)       # TODAY_OVERRIDE = 2026-09-18
    assert "9 PM is outside our booking hours (8 AM-8 PM)" in d["answer"] and "20 September" in d["answer"]
    assert "21:00" not in d["answer"]
    d = await chat(client, cid, "10 PM", sid)
    assert "10 PM is outside our booking hours" in d["answer"] and "between 8 AM and 8 PM" in d["answer"]
    assert "22:00" not in d["answer"] and "20 September" in d["answer"]

    d = await chat(client, cid, "10 AM", sid)
    assert d["action"]["type"] == "BOOK_APPOINTMENT"
    assert "at 10 AM" in d["answer"]
    res = d["action"]["result"]
    assert res["date"] == "2026-09-20" and res["time"] == "10:00" and "S-1031" in res["notes"]   # stored as HH:MM


async def test_24_hour_customers_get_24_hour_replies(client):
    d = await chat(client, "10011", "book a test drive on 25th September at 21:30", "s-24h")
    assert "21:30 is outside our booking hours (08:00-20:00)" in d["answer"]
