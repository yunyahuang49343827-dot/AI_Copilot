"""Runtime-only in-memory audit log management for agent workflows."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

AGENT_RUN_CREATED = "AGENT_RUN_CREATED"
APPROVAL_REQUIRED = "APPROVAL_REQUIRED"
APPROVAL_RECOMMENDED = "APPROVAL_RECOMMENDED"
READY_FOR_CONFIRMATION = "READY_FOR_CONFIRMATION"
MANUAL_REVIEW_REQUIRED = "MANUAL_REVIEW_REQUIRED"
CASE_BLOCKED_INSUFFICIENT_EVIDENCE = "CASE_BLOCKED_INSUFFICIENT_EVIDENCE"
APPROVAL_GRANTED = "APPROVAL_GRANTED"
APPROVAL_REJECTED = "APPROVAL_REJECTED"
APPROVAL_BLOCKED = "APPROVAL_BLOCKED"
CASE_CREATION_BLOCKED = "CASE_CREATION_BLOCKED"
CASE_CREATED = "CASE_CREATED"

CASE_CREATED_DIRECTLY = "CASE_CREATED_DIRECTLY"
LANGGRAPH_REVIEW_STARTED = "LANGGRAPH_REVIEW_STARTED"
LANGGRAPH_REVIEW_COMPLETED = "LANGGRAPH_REVIEW_COMPLETED"

_AUDIT_EVENTS: dict[str, dict[str, Any]] = {}
_NEXT_AUDIT_NUMBER = 1

_ALLOWED_METADATA_KEYS = {
    "status",
    "approval_required",
    "approval_recommended",
    "blocked",
    "case_created",
    "human_decision",
    "route",
    "allowed_action",
    "reason",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_audit_id() -> str:
    global _NEXT_AUDIT_NUMBER
    audit_id = f"AUDIT-{_NEXT_AUDIT_NUMBER:04d}"
    _NEXT_AUDIT_NUMBER += 1
    return audit_id


def _copy_event(event: dict[str, Any]) -> dict[str, Any]:
    copied = dict(event)
    copied["metadata"] = dict(event.get("metadata") or {})
    return copied


def sanitize_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not metadata:
        return {}
    return {key: metadata[key] for key in _ALLOWED_METADATA_KEYS if key in metadata}


def create_audit_event(
    event_type: str,
    message: str,
    actor: str | None = None,
    run_id: str | None = None,
    case_id: str | None = None,
    risk_level: str | None = None,
    risk_category: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    audit_id = _new_audit_id()
    event = {
        "audit_id": audit_id,
        "timestamp": _now(),
        "event_type": event_type,
        "actor": actor,
        "run_id": run_id,
        "case_id": case_id,
        "risk_level": risk_level,
        "risk_category": risk_category,
        "message": message,
        "metadata": sanitize_metadata(metadata),
    }
    _AUDIT_EVENTS[audit_id] = event
    return _copy_event(event)


def list_audit_events(
    limit: int | None = None,
    event_type: str | None = None,
    run_id: str | None = None,
    case_id: str | None = None,
) -> list[dict[str, Any]]:
    events = list(_AUDIT_EVENTS.values())
    if event_type:
        events = [event for event in events if event.get("event_type") == event_type]
    if run_id:
        events = [event for event in events if event.get("run_id") == run_id]
    if case_id:
        events = [event for event in events if event.get("case_id") == case_id]
    if limit is not None:
        events = events[-max(limit, 0) :]
    return [_copy_event(event) for event in events]


def get_audit_event(audit_id: str) -> dict[str, Any] | None:
    event = _AUDIT_EVENTS.get(audit_id)
    if event is None:
        return None
    return _copy_event(event)


def clear_audit_events() -> None:
    global _NEXT_AUDIT_NUMBER
    _AUDIT_EVENTS.clear()
    _NEXT_AUDIT_NUMBER = 1
