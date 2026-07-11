"""LangGraph orchestration layer for the controlled human-in-the-loop agent."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from src.agent.agent_state import AgentDecision, AgentState, DecisionTraceEntry, PendingAction
from src.agent.approval_gate import evaluate_approval_requirement

WorkflowFunc = Callable[[str, int], dict[str, Any]]


class LangGraphAgentState(TypedDict, total=False):
    query: str
    requester: str
    department: str
    top_k: int
    workflow_response: Dict[str, Any]
    risk_level: str
    risk_category: str
    decision: Dict[str, Any]
    pending_action: Optional[Dict[str, Any]]
    route: str
    trace: List[Dict[str, str]]
    case_created: bool
    case_id: Optional[str]
    final_agent_state: Optional[Dict[str, Any]]
    workflow_func: Optional[WorkflowFunc]


def append_trace(state: LangGraphAgentState, step: str, status: str, message: str) -> LangGraphAgentState:
    next_state = dict(state)
    trace = list(next_state.get("trace", []))
    trace.append({"step": step, "status": status, "message": message})
    next_state["trace"] = trace
    return next_state


def _normalize_top_k(value: Any) -> int:
    try:
        top_k = int(value)
    except (TypeError, ValueError):
        return 5
    return top_k if top_k > 0 else 5


def _decision_dict(decision: AgentDecision) -> dict[str, Any]:
    return asdict(decision)


def intake_node(state: LangGraphAgentState) -> LangGraphAgentState:
    next_state = dict(state)
    next_state["query"] = str(next_state.get("query", "") or "").strip()
    next_state["requester"] = str(next_state.get("requester", "") or "").strip() or "Demo User"
    next_state["department"] = str(next_state.get("department", "") or "").strip() or "Compliance"
    next_state["top_k"] = _normalize_top_k(next_state.get("top_k", 5))
    next_state["workflow_response"] = dict(next_state.get("workflow_response") or {})
    next_state["risk_level"] = str(next_state.get("risk_level", "") or "")
    next_state["risk_category"] = str(next_state.get("risk_category", "") or "")
    next_state["decision"] = dict(next_state.get("decision") or {})
    next_state["pending_action"] = next_state.get("pending_action")
    next_state["route"] = str(next_state.get("route", "") or "")
    next_state["trace"] = list(next_state.get("trace", []))
    next_state["case_created"] = False
    next_state["case_id"] = None
    next_state["final_agent_state"] = None
    return append_trace(next_state, "intake", "received", "LangGraph agent run initialized from user query.")


def workflow_node(state: LangGraphAgentState) -> LangGraphAgentState:
    next_state = dict(state)
    query = str(next_state.get("query", "") or "")
    top_k = _normalize_top_k(next_state.get("top_k", 5))
    workflow_func = next_state.get("workflow_func")
    if workflow_func is None:
        from src.agent.agent_tools import run_workflow_tool

        workflow_func = run_workflow_tool
    workflow_response = workflow_func(query, top_k)
    risk_level = str(workflow_response.get("risk_level", "") or "")
    risk_category = str(workflow_response.get("risk_category", "") or "")

    next_state["workflow_response"] = workflow_response
    next_state["risk_level"] = risk_level
    next_state["risk_category"] = risk_category
    next_state["workflow_func"] = None
    return append_trace(
        next_state,
        "workflow",
        "generated",
        f"Workflow advice generated with risk level: {risk_level or 'Unknown'}.",
    )


def approval_gate_node(state: LangGraphAgentState) -> LangGraphAgentState:
    next_state = dict(state)
    decision = evaluate_approval_requirement(
        str(next_state.get("risk_level", "") or ""),
        str(next_state.get("risk_category", "") or ""),
    )
    next_state["decision"] = _decision_dict(decision)
    return append_trace(next_state, "approval_gate", "evaluated", decision.reason)


def route_by_decision(state: LangGraphAgentState) -> str:
    decision = dict(state.get("decision") or {})
    if decision.get("blocked"):
        return "blocked"
    if decision.get("approval_required"):
        return "pending_approval"
    return "ready_for_confirmation"


def blocked_node(state: LangGraphAgentState) -> LangGraphAgentState:
    next_state = dict(state)
    decision = dict(next_state.get("decision") or {})
    next_state["pending_action"] = {
        "action_type": "manual_review",
        "description": "Manual review recommended because evidence is insufficient or risk is unknown.",
        "requires_approval": False,
        "blocked": True,
        "reason": decision.get("reason"),
    }
    next_state["route"] = "blocked"
    next_state["case_created"] = False
    next_state["case_id"] = None
    return append_trace(next_state, "pending_action", "blocked", "Case creation blocked; manual review required.")


def pending_approval_node(state: LangGraphAgentState) -> LangGraphAgentState:
    next_state = dict(state)
    decision = dict(next_state.get("decision") or {})
    next_state["pending_action"] = {
        "action_type": "create_mock_case",
        "description": "Create a mock compliance case after human approval.",
        "requires_approval": True,
        "blocked": False,
        "reason": decision.get("reason"),
    }
    next_state["route"] = "pending_approval"
    next_state["case_created"] = False
    next_state["case_id"] = None
    return append_trace(next_state, "pending_action", "prepared", "Mock case creation prepared pending approval.")


def ready_for_confirmation_node(state: LangGraphAgentState) -> LangGraphAgentState:
    next_state = dict(state)
    decision = dict(next_state.get("decision") or {})
    next_state["pending_action"] = {
        "action_type": "create_mock_case",
        "description": "Create a mock compliance case after positive confirmation.",
        "requires_approval": False,
        "blocked": False,
        "reason": decision.get("reason"),
    }
    next_state["route"] = "ready_for_confirmation"
    next_state["case_created"] = False
    next_state["case_id"] = None
    return append_trace(next_state, "pending_action", "prepared", "Mock case creation prepared for confirmation.")


def final_node(state: LangGraphAgentState) -> LangGraphAgentState:
    next_state = append_trace(state, "final", "completed", "LangGraph agent orchestration completed.")
    next_state["case_created"] = False
    next_state["case_id"] = None
    next_state["final_agent_state"] = graph_result_to_agent_state(next_state).to_dict()
    return next_state


def build_agent_graph():
    graph = StateGraph(LangGraphAgentState)
    graph.add_node("intake", intake_node)
    graph.add_node("workflow", workflow_node)
    graph.add_node("approval_gate", approval_gate_node)
    graph.add_node("blocked", blocked_node)
    graph.add_node("pending_approval", pending_approval_node)
    graph.add_node("ready_for_confirmation", ready_for_confirmation_node)
    graph.add_node("final", final_node)

    graph.add_edge(START, "intake")
    graph.add_edge("intake", "workflow")
    graph.add_edge("workflow", "approval_gate")
    graph.add_conditional_edges(
        "approval_gate",
        route_by_decision,
        {
            "blocked": "blocked",
            "pending_approval": "pending_approval",
            "ready_for_confirmation": "ready_for_confirmation",
        },
    )
    graph.add_edge("blocked", "final")
    graph.add_edge("pending_approval", "final")
    graph.add_edge("ready_for_confirmation", "final")
    graph.add_edge("final", END)
    return graph.compile()


def start_langgraph_agent_run(
    query: str,
    requester: str = "Demo User",
    department: str = "Compliance",
    top_k: int = 5,
    workflow_func: WorkflowFunc | None = None,
) -> dict[str, Any]:
    graph = build_agent_graph()
    initial_state: LangGraphAgentState = {
        "query": query,
        "requester": requester,
        "department": department,
        "top_k": top_k,
        "workflow_response": {},
        "risk_level": "",
        "risk_category": "",
        "decision": {},
        "pending_action": None,
        "route": "",
        "trace": [],
        "case_created": False,
        "case_id": None,
        "final_agent_state": None,
        "workflow_func": workflow_func,
    }
    return graph.invoke(initial_state)


def graph_result_to_agent_state(result: dict[str, Any]) -> AgentState:
    source = dict(result.get("final_agent_state") or result)
    decision_data = dict(source.get("decision") or {})
    pending_action_data = source.get("pending_action")
    trace_data = list(source.get("trace") or [])

    decision = AgentDecision(
        approval_required=bool(decision_data.get("approval_required", False)),
        approval_recommended=bool(decision_data.get("approval_recommended", False)),
        blocked=bool(decision_data.get("blocked", False)),
        allowed_action=decision_data.get("allowed_action"),
        reason=str(decision_data.get("reason", "") or ""),
    )
    pending_action = PendingAction(**pending_action_data) if pending_action_data else None
    trace = [DecisionTraceEntry(**entry) for entry in trace_data]

    return AgentState(
        query=str(source.get("query", "") or ""),
        requester=str(source.get("requester", "") or "") or "Demo User",
        department=str(source.get("department", "") or "") or "Compliance",
        top_k=_normalize_top_k(source.get("top_k", 5)),
        workflow_response=dict(source.get("workflow_response") or {}),
        risk_level=str(source.get("risk_level", "") or ""),
        risk_category=str(source.get("risk_category", "") or ""),
        pending_action=pending_action,
        decision=decision,
        case_created=False,
        case_id=None,
        trace=trace,
    )
