"""FastAPI backend with in-memory mock case management."""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException

from backend import services
from backend.agent_schemas import (
    AgentRunApproveRequest,
    AgentRunCreateRequest,
    AgentRunListResponse,
    AgentRunRejectRequest,
    AgentRunResponse,
)
from backend.audit_schemas import AuditEventResponse, AuditLogListResponse
from backend.schemas import (
    CaseCreateRequest,
    CaseResponse,
    CasesListResponse,
    HealthResponse,
    QAResponse,
    QueryRequest,
    WorkflowAdviceResponse,
)


app = FastAPI(
    title="WITS Governance Compliance Copilot API",
    description="Local FastAPI backend with in-memory mock case management for the WITS policy copilot portfolio demo.",
    version="0.8.0",
)


def service_unavailable(exc: services.RetrievalServiceError) -> HTTPException:
    return HTTPException(status_code=503, detail=str(exc))


@app.get("/health", response_model=HealthResponse)
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "WITS Governance Compliance Copilot API",
        "mode": "local-demo",
    }


@app.post("/qa", response_model=QAResponse)
def qa(request: QueryRequest) -> dict:
    try:
        return services.build_qa_response(request.query, top_k=request.top_k, use_llm=request.use_llm)
    except services.RetrievalServiceError as exc:
        raise service_unavailable(exc) from exc


@app.post("/workflow-advice", response_model=WorkflowAdviceResponse)
def workflow_advice(request: QueryRequest) -> dict:
    try:
        return services.build_workflow_response(request.query, top_k=request.top_k)
    except services.RetrievalServiceError as exc:
        raise service_unavailable(exc) from exc


@app.post("/cases", response_model=CaseResponse)
def create_case(request: CaseCreateRequest) -> dict:
    try:
        return services.create_case_from_workflow(
            request.query,
            requester=request.requester,
            department=request.department,
            top_k=request.top_k,
        )
    except services.RetrievalServiceError as exc:
        raise service_unavailable(exc) from exc


@app.get("/cases", response_model=CasesListResponse)
def list_cases() -> dict:
    return {"cases": services.list_cases()}


@app.get("/cases/{case_id}", response_model=CaseResponse)
def get_case(case_id: str) -> dict:
    case = services.get_case(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail=f"Case not found: {case_id}")
    return case


@app.post("/agent-runs", response_model=AgentRunResponse)
def create_agent_run(request: AgentRunCreateRequest) -> dict:
    try:
        return services.start_agent_run_service(
            request.query,
            requester=request.requester,
            department=request.department,
            top_k=request.top_k,
        )
    except services.RetrievalServiceError as exc:
        raise service_unavailable(exc) from exc


@app.get("/agent-runs", response_model=AgentRunListResponse)
def list_agent_runs() -> dict:
    return {"agent_runs": services.list_agent_runs_service()}


@app.get("/agent-runs/{run_id}", response_model=AgentRunResponse)
def get_agent_run(run_id: str) -> dict:
    run = services.get_agent_run_service(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Agent run not found: {run_id}")
    return run


@app.post("/agent-runs/{run_id}/approve", response_model=AgentRunResponse)
def approve_agent_run(run_id: str, request: AgentRunApproveRequest) -> dict:
    run = services.approve_agent_run_service(run_id, approver=request.approver)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Agent run not found: {run_id}")
    return run


@app.post("/agent-runs/{run_id}/reject", response_model=AgentRunResponse)
def reject_agent_run(run_id: str, request: AgentRunRejectRequest) -> dict:
    run = services.reject_agent_run_service(run_id, reviewer=request.reviewer, reason=request.reason)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Agent run not found: {run_id}")
    return run


@app.get("/audit-logs", response_model=AuditLogListResponse)
def list_audit_logs(
    event_type: Optional[str] = None,
    run_id: Optional[str] = None,
    case_id: Optional[str] = None,
    limit: Optional[int] = None,
) -> dict:
    return {
        "audit_events": services.list_audit_logs_service(
            event_type=event_type,
            run_id=run_id,
            case_id=case_id,
            limit=limit,
        )
    }


@app.get("/audit-logs/{audit_id}", response_model=AuditEventResponse)
def get_audit_log(audit_id: str) -> dict:
    event = services.get_audit_log_service(audit_id)
    if event is None:
        raise HTTPException(status_code=404, detail=f"Audit event not found: {audit_id}")
    return event
