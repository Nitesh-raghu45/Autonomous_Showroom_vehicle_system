import pytest

from app.agents.registry import AgentDomain
from app.agents.supervisor import _match_domain


@pytest.mark.parametrize("text", [
    "Does the Creta have air conditioning?",
    "Is air conditioning standard on the Swift?",
    "Book a test drive for the Creta, great service by the way",
    "Can I schedule a test drive tomorrow? Your customer service was helpful",
])
def test_sales_questions_do_not_reach_service_agent(text):
    assert _match_domain(text) is None


@pytest.mark.parametrize("text", [
    "My AC is not working",
    "the air conditioning stopped cooling, need a repair",
    "please check the AC",
    "book a car service for Monday",
    "I want to cancel my service appointment",
    "reschedule my service to Friday",
    "oil change next week",
])
def test_service_requests_still_reach_service_agent(text):
    assert _match_domain(text) == AgentDomain.SERVICE
