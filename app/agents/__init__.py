"""Multi-agent system: Supervisor Router dispatches to domain-specific agents.

Agents:
    - SalesAgent:      existing CRM / inventory / appointment logic (pass-through)
    - ServiceAgent:    vehicle service appointment booking, rescheduling, cancellation
    - ManagementAgent: senior management / business analysis (stub)
"""
from .registry import AgentDomain
from .supervisor import classify_and_dispatch

__all__ = ["AgentDomain", "classify_and_dispatch"]
