"""Thin tool wrappers for the controlled human-in-the-loop agent."""

from __future__ import annotations

from typing import Any

from backend import mock_case_store, services
from src.agent.agent_state import AgentState


def run_workflow_tool(query: str, top_k: int = 5) -> dict[str, Any]:
    return services.build_workflow_response(query, top_k=top_k)


def build_case_record_from_state(state: AgentState) -> dict[str, Any]:
    workflow = state.workflow_response
    return {
        "query": state.query,
        "requester": state.requester,
        "department": state.department,
        "risk_level": state.risk_level,
        "risk_category": state.risk_category,
        "risk_reasoning": workflow.get("risk_reasoning", ""),
        "recommended_steps": workflow.get("workflow_checklist", []),
        "recommended_next_steps": workflow.get("workflow_checklist", []),
        "citations": workflow.get("citations", []),
        "citation_summary": workflow.get("citations", []),
        "evidence_quality": workflow.get("evidence_quality", ""),
        "disclaimer": workflow.get("disclaimer", ""),
        "agent_decision": state.decision.to_dict() if hasattr(state.decision, "to_dict") else state.decision.__dict__,
        "human_decision": state.human_decision,
    }


def create_mock_case_tool(case_record: dict[str, Any]) -> dict[str, Any]:
    return mock_case_store.create_case(case_record)
