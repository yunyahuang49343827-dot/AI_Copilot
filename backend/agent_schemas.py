"""Pydantic schemas for human-in-the-loop agent run endpoints."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class AgentRunCreateRequest(BaseModel):
    query: str = Field(..., description="User question for an agent-mediated workflow run.")
    requester: str = Field(default="Demo User", description="Demo requester name or id.")
    department: str = Field(default="Compliance", description="Demo requester department.")
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("query cannot be blank")
        return value.strip()

    @field_validator("requester", "department")
    @classmethod
    def default_text_fields_when_blank(cls, value: str, info) -> str:
        stripped = (value or "").strip()
        if stripped:
            return stripped
        return "Demo User" if info.field_name == "requester" else "Compliance"


class AgentRunApproveRequest(BaseModel):
    approver: str = Field(default="Human Reviewer")

    @field_validator("approver")
    @classmethod
    def approver_default_when_blank(cls, value: str) -> str:
        return (value or "").strip() or "Human Reviewer"


class AgentRunRejectRequest(BaseModel):
    reviewer: str = Field(default="Human Reviewer")
    reason: Optional[str] = None

    @field_validator("reviewer")
    @classmethod
    def reviewer_default_when_blank(cls, value: str) -> str:
        return (value or "").strip() or "Human Reviewer"

    @field_validator("reason")
    @classmethod
    def blank_reason_to_none(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


class AgentRunResponse(BaseModel):
    run_id: str
    status: str
    created_at: str
    updated_at: str
    agent_state: dict[str, Any]
    approval_required: bool
    blocked: bool
    pending_action: Optional[dict[str, Any]]
    trace: list[dict[str, Any]]
    case_created: bool
    case_id: Optional[str]
    timing_metadata: Optional[dict[str, Any]] = None


class AgentRunListResponse(BaseModel):
    agent_runs: list[dict[str, Any]]
