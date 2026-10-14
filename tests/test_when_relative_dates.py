from datetime import date

import pytest

from app.router.when import parse_when

TODAY = date(2026, 10, 14)


@pytest.mark.parametrize("text,days", [
    ("tomorrow 10am", 1), ("tommorow 10am", 1), ("tomorow", 1), ("tommorrow", 1), ("tomarrow", 1),
    ("tmrw", 1), ("2moro at 11", 1), ("kal 11 baje", 1),
    ("day after tomorrow", 2), ("day after tommorow 9PM", 2), ("the day after tomorow", 2), ("parso", 2),
    ("today 11am", 0), ("todya 11am", 0), ("tonight 7pm", 0),
    ("yesterday 10am", -1), ("yesterdy 10am", -1),
])
def test_relative_days_tolerate_common_misspellings(text, days):
    assert (parse_when(text, TODAY).date - TODAY).days == days


def test_yesterday_is_rejected_as_past():
    assert "in the past" in parse_when("yesterdy at 10am", TODAY).date_error


def test_words_starting_with_tom_are_not_dates():
    assert parse_when("ask tom about it at 10", TODAY).date is None


# ---- replies use the customer's clock style ------------------------------------------------

from app.router.when import fmt_time  # noqa: E402


@pytest.mark.parametrize("h,m,ampm,expected", [
    (22, 0, True, "10 PM"), (10, 30, True, "10:30 AM"), (12, 0, True, "12 PM"), (0, 0, True, "12 AM"),
    (22, 0, False, "22:00"), (9, 5, None, "09:05"),
])
def test_fmt_time(h, m, ampm, expected):
    assert fmt_time(h, m, ampm) == expected


@pytest.mark.parametrize("text,ampm", [
    ("10 PM", True), ("9pm", True), ("10:30 a.m.", True), ("shaam 6 baje", True), ("6 o'clock", True),
    ("21:30", False), ("10:30", None), ("at 4", None),
])
def test_clock_style_is_detected(text, ampm):
    assert parse_when(text, TODAY).ampm is ampm


def test_out_of_hours_error_echoes_am_pm():
    assert parse_when("10 PM", TODAY).time_error == "10 PM is outside our booking hours (8 AM-8 PM)"
    assert parse_when("22:00", TODAY).time_error == "22:00 is outside our booking hours (08:00-20:00)"
