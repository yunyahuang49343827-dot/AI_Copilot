"""Approval rules for the controlled human-in-the-loop agent."""

from __future__ import annotations

from src.agent.agent_state import AgentDecision
from src.agent.risk_triage import RISK_HIGH, RISK_INSUFFICIENT, RISK_LOW, RISK_MEDIUM


def evaluate_approval_requirement(risk_level: str, risk_category: str | None = None) -> AgentDecision:
    level = (risk_level or "").strip()
    category_detail = f" for {risk_category}" if risk_category else ""

    if level == RISK_HIGH:
        return AgentDecision(
            approval_required=True,
            approval_recommended=True,
            blocked=False,
            allowed_action="create_mock_case_after_approval",
            reason=f"High-risk workflow{category_detail} requires human approval before mock case creation.",
        )

    if level == RISK_MEDIUM:
        return AgentDecision(
            approval_required=False,
            approval_recommended=True,
            blocked=False,
            allowed_action="create_mock_case",
            reason=f"Medium-risk workflow{category_detail} can create a mock case, with human review recommended.",
        )

    if level == RISK_LOW:
        return AgentDecision(
            approval_required=False,
            approval_recommended=False,
            blocked=False,
            allowed_action="create_mock_case",
            reason=f"Low-risk workflow{category_detail} does not require forced approval for mock case creation.",
        )

    if level == RISK_INSUFFICIENT:
        return AgentDecision(
            approval_required=False,
            approval_recommended=False,
            blocked=True,
            allowed_action=None,
            reason="Case creation is blocked because retrieved evidence is insufficient; manual review is required.",
        )

    return AgentDecision(
        approval_required=False,
        approval_recommended=False,
        blocked=True,
        allowed_action=None,
        reason="Risk level is unknown; manual review required.",
    )
