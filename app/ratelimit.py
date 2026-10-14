"""Shared slowapi limiter, kept outside main.py so routers can decorate endpoints
without a circular import. `default_limits` applies to every route without its own
decorator - but only because main.py installs SlowAPIMiddleware."""
from slowapi import Limiter
from slowapi.util import get_remote_address

from .config import settings

limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"], enabled=settings.rate_limit_enabled)
