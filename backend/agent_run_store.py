"""Runtime-only in-memory agent run management for the FastAPI backend."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.agent.agent_state import AgentState

_RUNS: dict[str, dict[str, Any]] = {}
_NEXT_RUN_NUMBER = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_run_id() -> str:
    global _NEXT_RUN_NUMBER
    run_id = f"AGENT-RUN-{_NEXT_RUN_NUMBER:04d}"
    _NEXT_RUN_NUMBER += 1
    return run_id


def derive_status(state: AgentState) -> str:
    if state.decision.blocked:
        return "Blocked"
    if state.case_created:
        return "CaseCreated"
    if state.human_decision == "approved":
        return "Approved"
    if state.human_decision == "rejected":
        return "Rejected"
    if state.decision.approval_required:
        return "PendingApproval"
    if state.decision.approval_recommended:
        return "ApprovalRecommended"
    return "ReadyForConfirmation"


def _full_response(record: dict[str, Any]) -> dict[str, Any]:
    state: AgentState = record["state"]
    status = derive_status(state)
    return {
        "run_id": record["run_id"],
        "status": status,
        "created_at": record["created_at"],
        "updated_at": record["updated_at"],
        "agent_state": state.to_dict(),
        "approval_required": state.decision.approval_required,
        "blocked": state.decision.blocked,
        "pending_action": state.pending_action.__dict__ if state.pending_action else None,
        "trace": [entry.__dict__ for entry in state.trace],
        "case_created": state.case_created,
        "case_id": state.case_id,
    }


def _summary(record: dict[str, Any]) -> dict[str, Any]:
    state: AgentState = record["state"]
    return {
        "run_id": record["run_id"],
        "status": derive_status(state),
        "query": state.query,
        "risk_level": state.risk_level,
        "risk_category": state.risk_category,
        "approval_required": state.decision.approval_required,
        "blocked": state.decision.blocked,
        "case_created": state.case_created,
        "case_id": state.case_id,
        "created_at": record["created_at"],
        "updated_at": record["updated_at"],
    }


def create_run(state: AgentState) -> dict[str, Any]:
    run_id = _new_run_id()
    timestamp = _now()
    _RUNS[run_id] = {
        "run_id": run_id,
        "created_at": timestamp,
        "updated_at": timestamp,
        "state": state,
    }
    return _full_response(_RUNS[run_id])


def get_run(run_id: str) -> dict[str, Any] | None:
    record = _RUNS.get(run_id)
    if record is None:
        return None
    return _full_response(record)


def get_run_state(run_id: str) -> AgentState | None:
    record = _RUNS.get(run_id)
    if record is None:
        return None
    return record["state"]


def list_runs() -> list[dict[str, Any]]:
    return [_summary(record) for record in _RUNS.values()]


def update_run(run_id: str, state: AgentState) -> dict[str, Any] | None:
    record = _RUNS.get(run_id)
    if record is None:
        return None
    record["state"] = state
    record["updated_at"] = _now()
    return _full_response(record)


def clear_runs() -> None:
    global _NEXT_RUN_NUMBER
    _RUNS.clear()
    _NEXT_RUN_NUMBER = 1
