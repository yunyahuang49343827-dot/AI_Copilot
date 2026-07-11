"""State containers for the controlled human-in-the-loop agent."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DecisionTraceEntry:
    step: str
    status: str
    message: str


@dataclass
class PendingAction:
    action_type: str
    description: str
    requires_approval: bool
    blocked: bool = False
    reason: str | None = None


@dataclass
class AgentDecision:
    approval_required: bool
    approval_recommended: bool
    blocked: bool
    allowed_action: str | None
    reason: str


@dataclass
class AgentState:
    query: str
    requester: str
    department: str
    top_k: int
    workflow_response: dict[str, Any]
    risk_level: str
    risk_category: str
    pending_action: PendingAction | None
    decision: AgentDecision
    human_decision: str | None = None
    case_created: bool = False
    case_id: str | None = None
    case_record: dict[str, Any] | None = None
    trace: list[DecisionTraceEntry] = field(default_factory=list)

    def add_trace(self, step: str, status: str, message: str) -> None:
        self.trace.append(DecisionTraceEntry(step=step, status=status, message=message))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
