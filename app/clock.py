"""Single source of 'now' / 'today'.

The dealership works in one timezone, so 'tomorrow' must be computed there - not in
the server's UTC. TODAY_OVERRIDE freezes the date for reproducible benchmarks.
"""
from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

from .config import settings

_TZ = ZoneInfo(settings.timezone)


def now() -> datetime:
    return datetime.now(_TZ)


def today() -> date:
    if settings.today_override:
        return date.fromisoformat(settings.today_override)
    return now().date()


def is_frozen() -> bool:
    return bool(settings.today_override)
