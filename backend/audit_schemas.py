"""Pydantic schemas for runtime audit log endpoints."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class AuditEventResponse(BaseModel):
    audit_id: str
    timestamp: str
    event_type: str
    actor: Optional[str]
    run_id: Optional[str]
    case_id: Optional[str]
    risk_level: Optional[str]
    risk_category: Optional[str]
    message: str
    metadata: dict[str, Any]


class AuditLogListResponse(BaseModel):
    audit_events: list[dict[str, Any]]
