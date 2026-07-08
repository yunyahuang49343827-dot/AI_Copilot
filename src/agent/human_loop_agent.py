"""Service-level orchestration for controlled human-in-the-loop case creation."""

from __future__ import annotations

from typing import Any, Callable

from src.agent.agent_state import AgentState, PendingAction
from src.agent.agent_tools import build_case_record_from_state, create_mock_case_tool, run_workflow_tool
from src.agent.approval_gate import evaluate_approval_requirement

WorkflowFunc = Callable[[str, int], dict[str, Any]]
CaseCreator = Callable[[dict[str, Any]], dict[str, Any]]


def _pending_action_for_decision(decision, risk_level: str) -> PendingAction:
    if decision.blocked:
        return PendingAction(
            action_type="manual_review",
            description="Manual review is required before any case creation.",
            requires_approval=False,
            blocked=True,
            reason=decision.reason,
        )
    if decision.approval_required:
        return PendingAction(
            action_type="create_mock_case",
            description=f"Create a mock compliance case for {risk_level} risk after human approval.",
            requires_approval=True,
            blocked=False,
            reason=decision.reason,
        )
    return PendingAction(
        action_type="create_mock_case",
        description="Create a mock compliance case from the workflow advice.",
        requires_approval=False,
        blocked=False,
        reason=decision.reason,
    )


def start_agent_run(
    query: str,
    requester: str = "Demo User",
    department: str = "Compliance",
    top_k: int = 5,
    workflow_func: WorkflowFunc | None = None,
) -> AgentState:
    workflow_runner = workflow_func or run_workflow_tool
    trace_query = (query or "").strip()
    workflow_response = workflow_runner(trace_query, top_k)
    risk_level = str(workflow_response.get("risk_level", "") or "")
    risk_category = str(workflow_response.get("risk_category", "") or "")
    decision = evaluate_approval_requirement(risk_level, risk_category)
    pending_action = _pending_action_for_decision(decision, risk_level)

    state = AgentState(
        query=trace_query,
        requester=(requester or "").strip() or "Demo User",
        department=(department or "").strip() or "Compliance",
        top_k=top_k,
        workflow_response=workflow_response,
        risk_level=risk_level,
        risk_category=risk_category,
        pending_action=pending_action,
        decision=decision,
    )
    state.add_trace("intake", "received", "Agent run initialized from user query.")
    state.add_trace("workflow_advice", "generated", f"Workflow advice generated with risk level: {risk_level or 'Unknown'}.")
    state.add_trace("approval_gate", "evaluated", decision.reason)
    if decision.blocked:
        state.add_trace("pending_action", "blocked", "Case creation blocked; manual review required.")
    else:
        state.add_trace("pending_action", "prepared", pending_action.description)
    return state


def approve_agent_action(
    state: AgentState,
    approver: str = "Human Reviewer",
    case_creator: CaseCreator | None = None,
) -> AgentState:
    if state.case_created:
        state.add_trace("approval", "skipped", f"Case already created by prior approval; reviewer: {approver}.")
        return state

    if state.decision.blocked:
        state.human_decision = "blocked"
        state.add_trace("approval", "blocked", "Approval cannot create a case because the decision is blocked.")
        return state

    if state.pending_action is None:
        state.human_decision = "no_action"
        state.add_trace("approval", "no_action", "No pending action was available for approval.")
        return state

    if state.pending_action.action_type != "create_mock_case" or state.decision.allowed_action is None:
        state.human_decision = "no_action"
        state.add_trace("approval", "no_action", "Pending action is not allowed for mock case creation.")
        return state

    state.human_decision = "approved"
    case_record = build_case_record_from_state(state)
    creator = case_creator or create_mock_case_tool
    created_case = creator(case_record)
    state.case_created = True
    state.case_id = created_case.get("case_id")
    state.case_record = created_case
    state.add_trace("approval", "approved", f"Mock case creation approved by {approver}.")
    state.add_trace("case_creation", "created", f"Mock case created: {state.case_id or 'unknown case id'}.")
    return state


def reject_agent_action(
    state: AgentState,
    reviewer: str = "Human Reviewer",
    reason: str | None = None,
) -> AgentState:
    state.human_decision = "rejected"
    state.case_created = False
    state.case_id = None
    message = f"Pending action rejected by {reviewer}."
    if reason:
        message += f" Reason: {reason}"
    state.add_trace("approval", "rejected", message)
    return state
