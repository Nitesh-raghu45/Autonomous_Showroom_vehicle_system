"""Rule-based scoring of benchmark answers (no LLM judge).

Only rows with an objective ground truth are scored; everything else is reported as
'unscored' so the accuracy number is never inflated.
"""
from __future__ import annotations

import json
import re
from typing import Any

# Dataset rows whose stored ground truth is inconsistent with the data / calendar.
KNOWN_DATASET_ISSUES = {
    "BQ-139": "expects 48 for 'how many makes' - 48 is the vehicle count; the catalogue has 9 makes",
    "BQ-150": "expects 6, but only 4 Maruti Suzuki/Hyundai vehicles are <= 10 lakhs at all (3 are hatchbacks)",
    "BQ-109": "dataset expects a primary-mobile change to be rejected; product decision since changed: allowed when the number is valid and unique",
    "BQ-073": "expects 2026-09-22 for 'next Monday' authored on Fri 2026-09-18, but that Monday is 2026-09-21 (09-22 is a Tuesday)",
}
QUOTE_TOTAL_Q = re.compile(r"total|grand|math|including|price quoted", re.I)


def _digits(x: Any) -> str:
    return re.sub(r"\D", "", str(x))


def score(q: dict[str, Any], answer: str, action: dict[str, Any] | None) -> tuple[bool | None, str]:
    """-> (ok | None if unscored, reason)."""
    cat, gt, exp, low = q["category"], q.get("ground_truth"), q.get("expected_action"), answer.lower()
    if q["id"] in KNOWN_DATASET_ISSUES:
        return None, "known dataset issue: " + KNOWN_DATASET_ISSUES[q["id"]]

    if isinstance(exp, dict):
        if not action or action.get("type") != exp["type"]:
            return False, f"expected action {exp['type']}, got {action and action.get('type')}"
        res = action.get("result") or {}
        if exp["type"] == "BOOK_APPOINTMENT":
            want = (exp.get("appointment_type"), exp.get("date"), f"{exp.get('hour', 0):02d}:{exp.get('minute', 0):02d}")
            got = (res.get("appointment_type"), res.get("date"), res.get("time"))
            return (want == got), f"want {want} got {got}"
        if exp["type"] == "UPDATE_INFO":
            cust = (res.get("record") or {}).get("customer", {})
            bad = [k for k in ("name", "email", "alt_mobile_no") if k in exp and cust.get(k) != exp[k]]
            return (not bad), f"mismatch on {bad}" if bad else "ok"
        if exp["type"] == "SEARCH_INVENTORY":
            bad = [k for k in ("make", "model", "vtype", "fuel", "min_price", "max_price") if k in exp and action.get(k) != exp[k]]
            if bad:
                return False, f"filter mismatch {bad}: got { {k: action.get(k) for k in bad} }"
        if isinstance(gt, str) and gt.isdigit():
            return (re.search(rf"\b{gt}\b", answer) is not None), f"count {gt}"
        return True, "ok"

    if gt in ("CLARIFY_OR_REJECT", "REJECT_OR_CLARIFY"):
        return (action is None), "must not execute an action"
    if gt == "NOT_IN_RECORD":
        return any(s in low for s in ("isn't on", "not on your record", "don't have", "no record")), "must admit the field is not on record"
    if gt == "REFUSE_OR_REDIRECT":
        return any(s in low for s in ("outside", "can't", "cannot", "only discuss", "can only", "not able")), "must refuse / redirect"
    if gt in (None, "REASON_FROM_CATALOG_TEXT") or (isinstance(gt, str) and gt.startswith("{")):
        return None, "no objective ground truth"
    if cat == "quotation_pricing":
        if QUOTE_TOTAL_Q.search(q["query"]):
            return (f"{int(gt):,}" in answer), f"grand total {gt}"
        return None, "dataset stores the grand total for every quotation row; question asks for another field"
    if cat == "customer_profile":
        return (str(gt).lower() in low), f"expected '{gt}'"
    if cat.startswith("inventory") and str(gt).isdigit():
        return (re.search(rf"\b{gt}\b", answer) is not None), f"count {gt}"
    return None, "unscored"
