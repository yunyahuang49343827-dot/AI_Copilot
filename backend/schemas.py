"""Pydantic schemas for the local FastAPI backend."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    query: str = Field(..., description="User question for grounded policy Q&A.")
    top_k: int = Field(default=5, ge=1, le=10)
    use_llm: bool = Field(default=False, description="Optional grounded LLM generation for Q&A only.")

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("query cannot be blank")
        return value.strip()


class CaseCreateRequest(QueryRequest):
    requester: str = Field(..., description="Demo requester name or id.")
    department: str = Field(..., description="Demo requester department.")

    @field_validator("requester", "department")
    @classmethod
    def field_not_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("field cannot be blank")
        return value.strip()


class CitationSummary(BaseModel):
    citation_id: str
    policy_name: str
    article: str
    article_title: str = ""
    pages: str
    source_file: str = ""
    chunk_id: str


class HealthResponse(BaseModel):
    status: str
    service: str
    mode: str


class QAResponse(BaseModel):
    query: str
    answer: str
    evidence_quality: str
    citations: list[CitationSummary]
    disclaimer: str
    generation_mode: str = "deterministic"
    llm_metadata: Optional[dict[str, Any]] = None
    timing_metadata: Optional[dict[str, Any]] = None


class WorkflowAdviceResponse(BaseModel):
    query: str
    grounded_answer: str
    risk_level: str
    risk_category: str
    risk_reasoning: str
    workflow_checklist: list[str]
    citations: list[CitationSummary]
    evidence_quality: str
    disclaimer: str
    timing_metadata: Optional[dict[str, Any]] = None


class CaseResponse(BaseModel):
    case_id: str
    query: str
    requester: str
    department: str
    risk_level: str
    risk_category: str
    status: str
    recommended_next_steps: list[str]
    created_at: str
    citation_summary: list[CitationSummary]
    evidence_quality: str
    disclaimer: str


class CasesListResponse(BaseModel):
    cases: list[CaseResponse]
