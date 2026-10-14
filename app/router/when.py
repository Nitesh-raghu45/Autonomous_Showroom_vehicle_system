"""Natural-language date / time extraction for appointment booking.

Handles: ISO dates, dd/mm[/yyyy], '25th September', 'Oct 2', 'the 26th', today/tonight,
tomorrow, day after tomorrow (incl. common misspellings like 'tommorow'), weekdays
('next Monday'), yesterday, and the Hinglish
'kal' / 'parso'. Times: 10am, 4:30 pm, 14:00, noon, '10 baje' (+ subah / shaam).

Anything vague ('Friday afternoon', 'whenever', 'next week') is reported as *vague*
so the caller can ask a clarifying question instead of guessing.
"""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, timedelta

MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4,
    "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8,
    "september": 9, "sept": 9, "sep": 9, "october": 10, "oct": 10, "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}
WEEKDAYS = {
    "monday": 0, "mon": 0, "tuesday": 1, "tue": 1, "tues": 1, "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thur": 3, "thurs": 3, "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5, "sunday": 6, "sun": 6,
}
_MONTH_RE = "|".join(sorted(MONTHS, key=len, reverse=True))
_WD_RE = "|".join(sorted(WEEKDAYS, key=len, reverse=True))

BUSINESS_OPEN, BUSINESS_CLOSE = 8, 20


def fmt_time(hour: int, minute: int = 0, ampm: bool | None = None) -> str:
    """Show a time the way the customer writes it: '10 PM' / '10:30 AM' if they used
    AM/PM, else 24-hour '22:00'."""
    if not ampm:
        return f"{hour:02d}:{minute:02d}"
    h12 = hour % 12 or 12
    return f"{h12}{f':{minute:02d}' if minute else ''} {'PM' if hour % 24 >= 12 else 'AM'}"


def hours_range(ampm: bool | None = None, sep: str = "-") -> str:
    return f"{fmt_time(BUSINESS_OPEN, 0, ampm)}{sep}{fmt_time(BUSINESS_CLOSE, 0, ampm)}"


@dataclass
class When:
    date: date | None = None
    date_error: str | None = None      # human-readable problem
    date_vague: bool = False
    hour: int | None = None
    minute: int = 0
    time_error: str | None = None
    time_vague: bool = False
    time_assumed: bool = False         # e.g. "at 4" -> 16:00 by business-hours convention
    ampm: bool | None = None           # True: customer wrote AM/PM, False: 24-hour ("14:00"), None: can't tell

    @property
    def complete(self) -> bool:
        return self.date is not None and self.hour is not None and not self.date_error and not self.time_error


# ---------------------------------------------------------------------------
# time
# ---------------------------------------------------------------------------

_T_COLON = re.compile(r"(?<![\d:/-])(\d{1,2}):(\d{2})\s*(a\.?m\.?|p\.?m\.?)?(?![\d:])")
_T_AMPM = re.compile(r"(?<![\d:/.-])(\d{1,2})\s*(a\.?m\.?|p\.?m\.?)(?![a-z])")
_T_BAJE = re.compile(r"(?<![\d:/.-])(\d{1,2})\s*baje")
_T_AT = re.compile(r"\b(?:at|@|by|around)\s+(\d{1,2})(?![\d:/-]|\s*(?:st|nd|rd|th|am|pm|a\.m|p\.m)\b)")
_T_OCLOCK = re.compile(r"(?<![\d:/.-])(\d{1,2})\s*o'?\s?clock")

_PM_WORDS = re.compile(r"\b(shaam|sham|raat|dopahar|evening|afternoon|night)\b")
_AM_WORDS = re.compile(r"\b(subah|savere|morning)\b")
_VAGUE_TIME = re.compile(r"\b(morning|afternoon|evening|night|whenever|any ?time|flexible|anytime|any slot|subah|shaam|dopahar)\b")


def _ampm(h: int, marker: str | None) -> int:
    if marker:
        pm = marker.lower().startswith("p")
        return (h % 12) + (12 if pm else 0)
    return h


def _parse_time(t: str, w: When) -> str:
    """Fill hour/minute on ``w``; return text with the matched span blanked out."""
    m = _T_COLON.search(t)
    if m:
        h, mi, mk = int(m.group(1)), int(m.group(2)), m.group(3)
        if mi > 59 or h > 24:
            w.time_error = f"{m.group(0).strip()} is not a valid time"
        else:
            w.hour, w.minute = _ampm(h, mk), mi
            w.ampm = True if mk else (False if h > 12 or h == 0 else None)
        return t[: m.start()] + " " + t[m.end():]

    m = _T_AMPM.search(t)
    if m:
        w.ampm = True
        h = int(m.group(1))
        if h < 1 or h > 12:
            w.time_error = f"{m.group(0).strip()} is not a valid time"
        else:
            w.hour, w.minute = _ampm(h, m.group(2)), 0
        return t[: m.start()] + " " + t[m.end():]

    m = _T_BAJE.search(t)
    if m:
        w.ampm = True                       # "10 baje" is a 12-hour way of speaking
        h = int(m.group(1))
        if _PM_WORDS.search(t) and h < 12:
            h += 12
        elif not _AM_WORDS.search(t) and 1 <= h <= 7:
            h += 12
            w.time_assumed = True
        w.hour, w.minute = h, 0
        return t[: m.start()] + " " + t[m.end():]

    m = _T_OCLOCK.search(t) or _T_AT.search(t)
    if m:
        if m.re is _T_OCLOCK:
            w.ampm = True
        h = int(m.group(1))
        if 1 <= h <= 7:
            h += 12
            w.time_assumed = True
        elif h == 24 or h == 0:
            h = 0
        w.hour, w.minute = h, 0
        return t[: m.start()] + " " + t[m.end():]

    if re.search(r"\b(noon|midday)\b", t):
        w.hour, w.minute = 12, 0
        return re.sub(r"\b(noon|midday)\b", " ", t)
    if re.search(r"\bmidnight\b", t):
        w.hour, w.minute = 0, 0
        return re.sub(r"\bmidnight\b", " ", t)

    if _VAGUE_TIME.search(t):
        w.time_vague = True
    return t


# ---------------------------------------------------------------------------
# date
# ---------------------------------------------------------------------------

def _mk(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def _dm_no_year(d: int, m: int, today: date) -> date | None:
    """day+month with no year: this year, else next year(s) until the date exists."""
    for y in range(today.year, today.year + 5):
        cand = _mk(y, m, d)
        if cand and cand >= today:
            return cand
    return None


def _set_date(w: When, d: int, m: int, y: int | None, today: date, raw: str) -> None:
    if not (1 <= m <= 12) or d < 1:
        w.date_error = f"“{raw.strip(' .,')}” is not a valid date"
        return
    if y is not None:
        if y < 100:
            y += 2000
        cand = _mk(y, m, d)
    else:
        cand = _dm_no_year(d, m, today)
        if cand is None:
            cand = _mk(today.year, m, d)
    if cand is None:
        w.date_error = f"“{raw.strip(' .,')}” is not a valid date"
    else:
        w.date = cand


# Relative days, tolerant of the common chat misspellings ("tommorow", "tomorow", "2moro",
# "todya", "yesterdy") - a missed date here is silently dropped and the booking stalls.
_TMRW = r"(?:t[ou]m+[ao]r+o?w|tmr+o?w?|2mor+o?w?)"
_DAY_AFTER = re.compile(rf"\bday\s+aft(?:er|r)\s+{_TMRW}\b|\bparso\b|\bovermorrow\b")
_TOMORROW = re.compile(rf"\b(?:{_TMRW}|kal)\b")
_TODAY = re.compile(r"\b(?:today|todya|toady|tday|tdy|2day|tonight|tonite|aaj)\b")
_YESTERDAY = re.compile(r"\b(?:y[ae]st[ae]?r?d[ae]?y|ystrdy)\b")


def _parse_date(t: str, w: When, today: date) -> None:
    # ISO
    m = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", t)
    if m:
        _set_date(w, int(m.group(3)), int(m.group(2)), int(m.group(1)), today, m.group(0))
        return
    # dd/mm(/yyyy)   dd-mm-yyyy   dd.mm.yyyy
    m = re.search(r"(?<![\d/])(\d{1,2})[/\-.](\d{1,2})(?:[/\-.](\d{2,4}))?(?![\d/])", t)
    if m:
        y = int(m.group(3)) if m.group(3) else None
        _set_date(w, int(m.group(1)), int(m.group(2)), y, today, m.group(0))
        return
    # 25th September [2026] / 25 sept / 25th of september
    m = re.search(rf"\b(\d{{1,2}})\s*(?:st|nd|rd|th)?\s*(?:of\s+)?({_MONTH_RE})\b\.?(?:[,\s]+(\d{{4}}))?", t)
    if m:
        y = int(m.group(3)) if m.group(3) else None
        _set_date(w, int(m.group(1)), MONTHS[m.group(2)], y, today, m.group(0))
        return
    # September 25[th] [2026]
    m = re.search(rf"\b({_MONTH_RE})\b\.?\s*(\d{{1,2}})\s*(?:st|nd|rd|th)?\b(?:[,\s]+(\d{{4}}))?", t)
    if m:
        y = int(m.group(3)) if m.group(3) else None
        _set_date(w, int(m.group(2)), MONTHS[m.group(1)], y, today, m.group(0))
        return

    if _DAY_AFTER.search(t):
        w.date = today + timedelta(days=2)
        return
    if _TOMORROW.search(t):
        w.date = today + timedelta(days=1)
        return
    if _TODAY.search(t):
        w.date = today
        return
    if _YESTERDAY.search(t):
        w.date = today - timedelta(days=1)
        return

    m = re.search(rf"\b(next|this|coming|upcoming)?\s*({_WD_RE})\b", t)
    if m and not re.search(rf"\b({_WD_RE})\s*(?:st|nd|rd|th)\b", t):
        target = WEEKDAYS[m.group(2)]
        delta = (target - today.weekday()) % 7
        if m.group(1) == "this":
            w.date = today + timedelta(days=delta)
        else:
            w.date = today + timedelta(days=delta or 7)
        return

    # bare day-of-month: "the 26th", "on 26th"
    m = re.search(r"\b(?:the\s+)?(\d{1,2})\s*(?:st|nd|rd|th)\b", t)
    if m:
        d = int(m.group(1))
        dim = calendar.monthrange(today.year, today.month)[1]
        if d < 1 or d > 31:
            w.date_error = f"“{m.group(0).strip()}” is not a valid day of the month"
        elif d >= today.day and d <= dim:
            w.date = date(today.year, today.month, d)
        else:
            ny, nm = (today.year + (today.month == 12), 1 if today.month == 12 else today.month + 1)
            cand = _mk(ny, nm, d)
            if cand:
                w.date = cand
            else:
                w.date_error = f"“{m.group(0).strip()}” is not a valid day of the month"
        return

    if re.search(r"\b(next week|this week|next month|whenever|any ?day|any date|flexible|some ?day)\b", t):
        w.date_vague = True


def parse_when(text: str, today: date) -> When:
    t = " " + text.lower() + " "
    w = When()
    t = _parse_time(t, w)
    _parse_date(t, w, today)
    if w.hour is not None and not w.time_error:
        if not (BUSINESS_OPEN <= w.hour <= BUSINESS_CLOSE):
            w.time_error = (
                f"{fmt_time(w.hour, w.minute, w.ampm)} is outside our booking hours "
                f"({hours_range(w.ampm)})"
            )
    if w.date is not None and not w.date_error and w.date < today:
        w.date_error = f"{w.date.isoformat()} is in the past"
    return w
