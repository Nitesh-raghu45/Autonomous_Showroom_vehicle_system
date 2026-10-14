"""Slot extraction helpers (email, mobile, names, appointment type, inventory filters)."""
from __future__ import annotations

import re
from typing import Any

from .. import inventory

EMAIL_OK = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)+")
# (?<!\S): only start at a word boundary - same leftmost match, but linear instead of
# quadratic on a long unbroken run (a 2000-char token used to take ~70 ms here).
_EMAIL_LOOSE = re.compile(r"(?<!\S)(?:\S+@\S+|\S+\s*[\(\[]\s*at\s*[\)\]]\s*\S+)", re.I)

NAME_TOKEN = re.compile(r"[A-Za-z][A-Za-z.'\-]*")
NAME_STOP = {
    "mobile", "phone", "number", "email", "e-mail", "mail", "contact", "interested", "and", "is", "no",
    "with", "title", "name", "new", "used", "my", "i", "for", "in", "again", "now", "please", "here", "as",
    "a", "an", "the", "first", "form", "to", "at", "on", "me", "wait", "but", "alt", "alternate", "sign",
}


NAME_STOP_LITE = {"mobile", "phone", "number", "email", "e-mail", "mail", "contact", "interested", "and", "title", "alt", "alternate", "wait"}


def find_email(text: str) -> tuple[str | None, str | None]:
    """-> (valid_email, invalid_attempt)."""
    m = EMAIL_OK.search(text)
    if m:
        return m.group(0), None
    m = _EMAIL_LOOSE.search(text)
    if m:
        return None, m.group(0).strip(" ,.;")
    return None, None


_DATEISH = re.compile(r"\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}(?:/\d{2,4})?")
_NUM = re.compile(r"\+?\d[\d\s\-]{3,}\d")


def find_numbers(text: str) -> list[str]:
    """Digit runs that look like phone numbers (>=5 digits), normalised. Dates are ignored."""
    t = _DATEISH.sub(" ", text)
    out = []
    for m in _NUM.finditer(t):
        digits = re.sub(r"\D", "", m.group(0))
        if len(digits) == 12 and digits.startswith("91"):
            digits = digits[2:]
        if len(digits) >= 5:
            out.append(digits)
    return out


def take_name(s: str, strict: bool = True) -> str:
    out: list[str] = []
    for tok in s.split():
        core = tok.rstrip(",;:!?")
        ends = core != tok
        core2 = core.rstrip(".")
        if not NAME_TOKEN.fullmatch(core) or core2.lower() in (NAME_STOP if strict else NAME_STOP_LITE):
            break
        out.append(core)
        if ends or core.endswith(".") and len(core2) > 1 or len(out) == 4:
            break
    return " ".join(out).rstrip(".,")


def name_after(text: str, pattern: str, strict: bool = True) -> str | None:
    m = re.search(pattern, text, re.I)
    if not m:
        return None
    n = take_name(text[m.end():], strict)
    return n or None


APPT_TYPES = [
    ("TEST_DRIVE", r"test[\s-]?drive|\bdrive\b|\bdemo\b"),
    ("CALLBACK", r"call[\s-]?back|call me|phone call|\bcall\b"),
    ("ADVANCE_PAYMENT", r"advance|down payment|token|\bpayment\b"),
    ("DOCUMENT_SUBMISSION", r"document|docs?\b|paperwork|submit"),
    ("OTHER", r"follow[\s-]?up|general|meeting|\bother\b|\bvisit\b"),
]


def appt_type(low: str) -> str | None:
    for name, pat in APPT_TYPES:
        if re.search(pat, low):
            return name
    return None


def notes_after(text: str) -> str | None:
    m = re.search(r"\bnotes?\s*[:\-]\s*(.+?)\s*$", text, re.I)
    return m.group(1).strip(" .") if m else None


# ---------------------------------------------------------------------------
# inventory filters
# ---------------------------------------------------------------------------

_AMBIGUOUS_MODELS = {"city", "punch", "safari", "venue", "amaze", "swift", "harrier"}
_FUELS = [
    ("Electric", r"\belectric\b|\bevs?\b|\bbev\b|battery"),
    ("Petrol", r"\bpetrols?\b"),
    ("Diesel", r"\bdiesels?\b"),
    ("CNG", r"\bcng\b"),
    ("Hybrid", r"\bhybrids?\b|\bhev\b"),
]
_BODY = [
    ("suv", r"\bsuvs?\b|\bcrossovers?\b"),
    ("hatchback", r"\bhatch(?:back)?s?\b|\bhatchbacks\b"),
    ("sedan", r"\bsedans?\b"),
    ("mpv", r"\bmpvs?\b|\bminivans?\b|\bpeople carriers?\b"),
]
_UNIT = r"(lakhs?|lacs?|crores?|cr|l)\b"
_N = r"(\d+(?:[.,]\d+)?)"
# a pattern that STARTS with a number must start at the number's first digit; otherwise a
# long digit run is retried from every position (quadratic: 2000 digits took ~460 ms)
_N0 = r"(?<!\d)" + _N


def _amount(num: str, unit: str | None) -> int:
    v = float(num.replace(",", ""))
    u = (unit or "").lower()
    if u.startswith("cr"):
        return int(v * 10_000_000)
    if u in ("l", "lakh", "lakhs", "lac", "lacs"):
        return int(v * 100_000)
    return int(v)


def parse_price(low: str) -> tuple[int | None, int | None]:
    # range: "between 40 and 90 lakhs", "15-20 lakhs", "25 to 35 lakh"
    m = re.search(rf"{_N0}\s*(?:{_UNIT})?\s*(?:-|–|to|and)\s*{_N}\s*{_UNIT}", low)
    if m and re.search(r"between|budget|range|from|-|–|\bto\b", low):
        unit = m.group(4)
        return _amount(m.group(1), m.group(2) or unit), _amount(m.group(3), unit)
    m = re.search(rf"(?:under|below|less than|upto|up to|within|max(?:imum)?|not more than|cheaper than|budget (?:of|is)?)\s*(?:rs\.?|inr|₹)?\s*{_N}\s*(?:{_UNIT})?", low)
    if m and not re.match(r"\d+\s*(?:km|kms|kilometres|kilometers)", low[m.start(1):]):
        tail = low[m.end(1):m.end(1) + 12]
        if not re.match(r"\s*(?:k\s*)?(?:km|kms)\b", tail):
            return None, _amount(m.group(1), m.group(2))
    m = re.search(rf"(?:above|over|more than|at least|min(?:imum)?|starting (?:from|at)|from)\s*(?:rs\.?|inr|₹)?\s*{_N}\s*{_UNIT}", low)
    if m:
        return _amount(m.group(1), m.group(2)), None
    # postfix order: "50 lacs above", "20 lakh and above", "1 cr+"
    m = re.search(rf"{_N0}\s*{_UNIT}\s*(?:and\s+)?(?:above|over|\+|plus)\b", low)
    if m:
        return _amount(m.group(1), m.group(2)), None
    m = re.search(rf"budget (?:of|is|around)?\s*(?:rs\.?|inr|₹)?\s*{_N}\s*{_UNIT}", low)
    if m:
        return None, _amount(m.group(1), m.group(2))
    return None, None


def parse_km(low: str) -> int | None:
    m = re.search(r"(?:under|below|less than|upto|up to|within|max)\s*([\d,]+)\s*(k)?\s*(?:km|kms|kilometers|kilometres)", low)
    if m:
        v = int(m.group(1).replace(",", ""))
        return v * 1000 if m.group(2) else v
    return None


_WORD_NUM = {"four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8}
_YEAR = r"(?<![\d\-/.])((?:19[89]|20[0-3])\d)(?![\d\-/]|\s*(?:km|kms|kilometers|kilometres|lakhs?|lacs?|crores?|cr|l)\b)"


def parse_year(low: str) -> tuple[list[int] | None, int | None]:
    """-> (exact_years, min_year). Dates (2026-10-10) and amounts/odometer readings are ignored."""
    m = re.search(rf"\b(?:after|newer than|later than)\s+{_YEAR}", low)
    if m:
        return None, int(m.group(1)) + 1
    m = re.search(rf"\b(?:since|from)\s+{_YEAR}", low)
    if m:
        return None, int(m.group(1))
    m = re.search(rf"{_YEAR}\s*(?:\+|onwards?|or (?:newer|later|above)|and (?:newer|later|above))", low)
    if m:
        return None, int(m.group(1))
    found = [int(y) for y in re.findall(_YEAR, low)]
    return (list(dict.fromkeys(found)) or None), None


def inventory_filters(text: str) -> dict[str, Any]:
    cat = inventory.catalog()
    low = text.lower()
    f: dict[str, Any] = {}

    makes = cat.find_makes(low)
    models = []
    for mk, mo in cat.find_models(low):
        if mo.lower() in _AMBIGUOUS_MODELS and mk not in makes and mo.lower() not in low:
            continue
        models.append((mk, mo))
    if models:
        f["models"] = [mo for _, mo in models]
        for mk, _ in models:
            if mk not in makes:
                makes.append(mk)
    if makes:
        f["makes"] = makes

    scrub = re.sub(r"\bnew (customer|registration|account|enquiry)\b", " ", low)
    used = bool(re.search(r"\bused\b|pre[\s-]?owned|second[\s-]?hand", scrub))
    new = bool(re.search(r"\bnew\b|brand new", scrub))
    if used and not new:
        f["vtype"] = "used"
    elif new and not used:
        f["vtype"] = "new"

    fuels = [name for name, pat in _FUELS if re.search(pat, low)]
    if len(fuels) == 1:
        f["fuel"] = fuels[0]
    elif fuels:
        f["fuels"] = fuels
    for name, pat in _BODY:
        if re.search(pat, low):
            f["body"] = name
            break
    ms = re.search(r"\b(\d|four|five|six|seven|eight)[\s-]?seat(?:er|ers|s)?\b", low)
    if ms:
        n = _WORD_NUM.get(ms.group(1)) or int(ms.group(1))
        f["min_seats"] = n
        if n != 7:                                  # "5 seater" means exactly 5; "7 seater" keeps its >=7 meaning
            f["max_seats"] = n

    # ---- variant / year / colour (the "exact vehicle" dimensions) -------------------------
    if f.get("models"):
        variants = cat.find_variants(low, f["models"])
        if variants:
            f["variants"] = list(dict.fromkeys(v for _, _, v in variants))
    years, min_year = parse_year(low)
    if years:
        f["years"] = years
    if min_year is not None:
        f["min_year"] = min_year
    colors = cat.find_colors(low)
    if colors:
        f["colors"] = colors
    if re.search(r"\bautomatic\b|\bamt\b|\bauto\b", low):
        f["transmission"] = "Automatic"
    elif re.search(r"\bmanual\b", low):
        f["transmission"] = "Manual"

    lo, hi = parse_price(low)
    if lo is not None:
        f["min_price"] = lo
    if hi is not None:
        f["max_price"] = hi
    km = parse_km(low)
    if km is not None:
        f["max_km"] = km
    if re.search(r"cheapest|lowest (price|cost|priced)|least expensive|most affordable|lowest priced|low(est)? budget|minimum (price|cost)|\bmin price\b", low):
        f["sort"] = "price_asc"
    elif re.search(r"most expensive|costliest|priciest|highest (price|cost|priced)|highest priced|most premium|maximum (price|cost)|\bmax price\b", low):
        f["sort"] = "price_desc"
    elif re.search(r"lowest mileage|least driven|lowest km", low):
        f["sort"] = "km_asc"
    elif re.search(r"\bnewest\b|\bmost recent\b|\blatest (?:model|car|vehicle|one|year)", low):
        f["sort"] = "year_desc"
    elif re.search(r"\boldest\b", low):
        f["sort"] = "year_asc"

    # "5 cheapest cars" / "cheapest 3" / "3 most expensive SUVs"
    _sup = r"(?:cheapest|costliest|priciest|most expensive|least expensive|most affordable|lowest priced)"
    _not_unit = r"(?!\s*(?:lakhs?|lacs?|crores?|cr|l|km|kms|seat))"
    mm = re.search(rf"\b(\d{{1,2}}){_not_unit}\s+{_sup}\b|\b{_sup}\s+(\d{{1,2}})\b{_not_unit}", low)
    if mm:
        n = int(next(g for g in mm.groups() if g))
        if 1 < n <= 20:
            f["limit"] = n

    m = re.search(r"\btop\s*(\d{1,2})\b|\bfirst\s*(\d{1,2})\b|\b(\d{1,2})\s*(?:best|top)\b", low)
    if m:
        n = int(next(g for g in m.groups() if g))
        if 1 < n <= 20:
            f["limit"] = n
            # "top N" with no explicit direction named implies "best"/highest-priced,
            # matching how people actually use the phrase - but never override a sort
            # the person already named (e.g. "top 3 cheapest").
            f.setdefault("sort", "price_desc")
    return f


OTHER_BRANDS = (
    "ferrari|tesla|mercedes|merc|benz|audi|volkswagen|vw|skoda|renault|nissan|ford|jeep|lamborghini|porsche|volvo|jaguar|"
    "land rover|range rover|lexus|byd|citroen|fiat|mini cooper|rolls[- ]royce|bentley|maserati|mitsubishi|subaru|chevrolet|"
    "datsun|isuzu|force motors|lotus|aston martin|bugatti|mclaren|ford|peugeot|opel|suzuki motorcycles|harley|royal enfield|bajaj|tvs|hero"
)
_OTHER_RE = re.compile(rf"\b({OTHER_BRANDS})\b")


def other_brand(low: str) -> str | None:
    """A brand we recognise as a car/bike make but do NOT stock (only when no catalogue make is named)."""
    m = _OTHER_RE.search(low)
    return m.group(1) if m else None