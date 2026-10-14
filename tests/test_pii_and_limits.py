import dataclasses
import logging

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from app import json_knowledge, service_main

RECORD = {"customer": {"customer_id": "10001", "name": "Amit Verma", "mobile_no": "9960013389",
                       "alt_mobile_no": None, "email": "customer10001@example.com"}}


def test_contact_fields_are_masked_for_unrelated_questions():
    ctx = json_knowledge.build_json_context(question="when was my enquiry created?", cid="10001", record=RECORD, appts=[])
    assert "9960013389" not in ctx and "customer10001@example.com" not in ctx
    assert "******3389" in ctx and "Amit Verma" in ctx


def test_contact_fields_are_sent_when_the_question_is_about_them():
    ctx = json_knowledge.build_json_context(question="what is my mobile number?", cid="10001", record=RECORD, appts=[])
    assert "9960013389" in ctx


def test_log_formatter_redacts_mobiles_and_emails():
    from app.main import _PiiRedactingFormatter
    rec = logging.LogRecord("t", logging.INFO, __file__, 1, "q=%r num=%s", ("call me on +91 9960013389", "customer10001@example.com"), None)
    out = _PiiRedactingFormatter("%(message)s").format(rec)
    assert "9960013389" not in out and "******3389" in out
    assert "customer10001@example.com" not in out and "c***@example.com" in out


def _req(headers: dict[str, str]) -> Request:
    return Request({"type": "http", "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()], "client": ("1.2.3.4", 1)})


def test_slot_webhook_requires_secret_when_configured(monkeypatch):
    monkeypatch.setattr(service_main, "settings", dataclasses.replace(service_main.settings, service_webhook_secret="s3cret"))
    with pytest.raises(HTTPException) as exc:
        service_main._require_webhook_secret(_req({}))
    assert exc.value.status_code == 401
    service_main._require_webhook_secret(_req({"X-Webhook-Secret": "s3cret"}))


def test_default_rate_limit_middleware_is_installed():
    from slowapi.middleware import SlowAPIMiddleware
    from app.main import app
    assert any(m.cls is SlowAPIMiddleware for m in app.user_middleware)
