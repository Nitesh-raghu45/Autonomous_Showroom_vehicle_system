"""Safety net for the LLM planner path (see planner.py / prompting.py).

Small/local models (this project defaults to a 4B, q4-quantized Ollama model -
see config.py:ollama_model) sometimes fail to engage with an ambiguous, vague,
or emotional message and instead fall back to reciting whatever fact was most
salient earlier in the conversation (typically something from the CUSTOMER
RECORD block, e.g. the customer's quotation). The result is a reply that is
byte-for-byte identical to the assistant's previous turn, even though the new
question shares no real content with the question that produced that answer.

This module does not try to understand *why* the input was confusing - it only
detects the degenerate symptom (verbatim self-repetition across unrelated
questions) so `service.py` can swap in an honest "I didn't understand that"
reply instead of silently resending stale information.
"""
from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9]+")

# Common function words carry no topic information, so they're excluded before
# comparing whether two questions are "about the same thing".
_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "do", "does", "did", "i", "you",
    "your", "yours", "my", "mine", "me", "to", "of", "for", "and", "or", "what",
    "why", "how", "who", "when", "which", "please", "this", "that", "it", "on",
    "in", "at", "be", "been", "have", "has", "had", "not", "no", "am", "will",
    "would", "can", "could", "just", "like", "want", "wanted", "know", "tell",
}


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall((text or "").lower()) if w not in _STOP and len(w) > 2}


def is_degenerate_repeat(
    question: str,
    previous_question: str | None,
    previous_answer: str | None,
    new_answer: str,
) -> bool:
    """True when `new_answer` is (near-)identical to `previous_answer` while the
    current `question` shares no real vocabulary with `previous_question`.

    A genuine repeat question ("what was my quotation" asked twice) is allowed
    through, since the two questions overlap in content words. Only the
    context-blind case - a new, unrelated question met with yesterday's answer,
    word for word - is flagged.
    """
    if not previous_answer or not previous_question or not new_answer:
        return False
    prev_norm = re.sub(r"\s+", " ", previous_answer.strip().lower())
    new_norm = re.sub(r"\s+", " ", new_answer.strip().lower())
    if not prev_norm or prev_norm != new_norm:
        return False
    if _content_words(question) & _content_words(previous_question):
        return False
    return True


GENERIC_CLARIFY = (
    "Sorry, I don't think I understood that message. Could you tell me what "
    "you'd like to know or do - your enquiry, quotation, appointments, or our "
    "vehicle stock?"
)