"""Service layer for the local FastAPI backend."""

from __future__ import annotations

from typing import Any

from backend import agent_run_store, audit_log_store, mock_case_store
from backend.performance import elapsed_ms, start_timer
from src.agent.human_loop_agent import approve_agent_action, reject_agent_action, start_agent_run
from src.agent.workflow_advisor import (
    DEMO_TRIAGE_NOTE,
    evidence_note,
    prepare_workflow_context,
    render_grounded_answer_section,
)
from src.generation.answer_builder import (
    build_direct_answer,
    build_evidence_lines,
    build_explanation,
    cited_results,
    hydrate_results,
    select_body_evidence,
)
from src.generation.grounded_llm_answer import generate_grounded_qa_answer
from src.generation.llm_client import LLMClient, create_deepseek_client_from_env
from src.guardrails.disclaimers import DEMO_DISCLAIMER
from src.guardrails.evidence_checks import assess_evidence
from src.retrieval.hybrid_search import hybrid_search


API_DISCLAIMER = f"{DEMO_DISCLAIMER} {DEMO_TRIAGE_NOTE}"


class RetrievalServiceError(RuntimeError):
    """Raised when local retrieval indexes or services are unavailable."""


def _safe_create_audit_event(
    event_type: str,
    message: str,
    actor: str | None = None,
    run_id: str | None = None,
    case_id: str | None = None,
    risk_level: str | None = None,
    risk_category: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    try:
        return audit_log_store.create_audit_event(
            event_type,
            message,
            actor=actor,
            run_id=run_id,
            case_id=case_id,
            risk_level=risk_level,
            risk_category=risk_category,
            metadata=metadata,
        )
    except Exception:
        return None


def _agent_audit_metadata(response: dict[str, Any]) -> dict[str, Any]:
    state = response.get("agent_state") or {}
    decision = state.get("decision") or {}
    return {
        "status": response.get("status"),
        "approval_required": response.get("approval_required"),
        "approval_recommended": decision.get("approval_recommended"),
        "blocked": response.get("blocked"),
        "case_created": response.get("case_created"),
        "human_decision": state.get("human_decision"),
        "allowed_action": decision.get("allowed_action"),
    }


def _agent_risk_fields(response: dict[str, Any]) -> tuple[str | None, str | None]:
    state = response.get("agent_state") or {}
    return state.get("risk_level"), state.get("risk_category")


def _log_agent_start_events(response: dict[str, Any], actor: str) -> None:
    risk_level, risk_category = _agent_risk_fields(response)
    run_id = response.get("run_id")
    metadata = _agent_audit_metadata(response)
    _safe_create_audit_event(
        audit_log_store.AGENT_RUN_CREATED,
        "Agent run created.",
        actor=actor,
        run_id=run_id,
        risk_level=risk_level,
        risk_category=risk_category,
        metadata=metadata,
    )
    if response.get("blocked") or risk_level == "Insufficient Evidence":
        _safe_create_audit_event(
            audit_log_store.CASE_BLOCKED_INSUFFICIENT_EVIDENCE,
            "Agent run blocked because evidence was insufficient or manual review is required.",
            actor=actor,
            run_id=run_id,
            risk_level=risk_level,
            risk_category=risk_category,
            metadata=metadata,
        )
    elif response.get("approval_required"):
        _safe_create_audit_event(
            audit_log_store.APPROVAL_REQUIRED,
            "Human approval is required before mock case creation.",
            actor=actor,
            run_id=run_id,
            risk_level=risk_level,
            risk_category=risk_category,
            metadata=metadata,
        )
    elif metadata.get("approval_recommended") or response.get("status") == "ApprovalRecommended":
        _safe_create_audit_event(
            audit_log_store.APPROVAL_RECOMMENDED,
            "Human review is recommended before mock case creation.",
            actor=actor,
            run_id=run_id,
            risk_level=risk_level,
            risk_category=risk_category,
            metadata=metadata,
        )
    else:
        _safe_create_audit_event(
            audit_log_store.READY_FOR_CONFIRMATION,
            "Agent run is ready for positive confirmation.",
            actor=actor,
            run_id=run_id,
            risk_level=risk_level,
            risk_category=risk_category,
            metadata=metadata,
        )


def _run_retrieval(query: str, top_k: int) -> list[dict[str, Any]]:
    try:
        return hybrid_search(query, top_k=top_k)
    except FileNotFoundError as exc:
        raise RetrievalServiceError(str(exc)) from exc
    except Exception as exc:
        raise RetrievalServiceError("Local retrieval service is unavailable. Confirm Chroma and BM25 indexes are built.") from exc


def citation_summary(citation_rows: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for label, result in citation_rows:
        page_start = result.get("page_start", "")
        page_end = result.get("page_end", "")
        summaries.append(
            {
                "citation_id": label,
                "policy_name": str(result.get("policy_name", "")),
                "article": str(result.get("article", "")),
                "article_title": str(result.get("article_title", "")),
                "pages": f"{page_start}-{page_end}",
                "source_file": str(result.get("source_file", "")),
                "chunk_id": str(result.get("chunk_id", "")),
            }
        )
    return summaries


def _llm_evidence_rows(rows: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    evidence_rows: list[dict[str, Any]] = []
    for label, result in rows:
        row = dict(result)
        row["citation_label"] = label
        evidence_rows.append(row)
    return evidence_rows


def build_qa_response(
    query: str,
    top_k: int = 5,
    retrieved_results: list[dict[str, Any]] | None = None,
    use_llm: bool = False,
    llm_client: LLMClient | None = None,
) -> dict[str, Any]:
    total_start = start_timer()
    retrieval_start = start_timer()
    results = retrieved_results if retrieved_results is not None else _run_retrieval(query, top_k)
    retrieval_time_ms = elapsed_ms(retrieval_start)

    answer_start = start_timer()
    hydrated = hydrate_results(results)
    citation_rows = cited_results(hydrated)
    body_rows = select_body_evidence(query, citation_rows)
    assessment = assess_evidence(query, [result for _, result in body_rows or citation_rows])
    direct = build_direct_answer(query, body_rows, assessment.is_sufficient)
    explanation = build_explanation(body_rows, assessment.is_sufficient)
    evidence_lines = build_evidence_lines(body_rows)
    answer = render_grounded_answer_section(direct, explanation, evidence_lines)
    evidence_quality = evidence_note(assessment)
    generation_mode = "deterministic"
    llm_metadata: dict[str, Any] | None = {"provider": "deepseek", "used_llm": False}

    if use_llm:
        evidence_rows = _llm_evidence_rows(body_rows or citation_rows)
        if not assessment.is_sufficient:
            llm_result = {
                "answer": answer,
                "generation_mode": "llm_fallback",
                "llm_metadata": {
                    "provider": "deepseek",
                    "used_llm": False,
                    "fallback_reason": "insufficient_evidence",
                    "evidence_count": len(evidence_rows),
                },
            }
        else:
            client = llm_client if llm_client is not None else create_deepseek_client_from_env()
            llm_result = generate_grounded_qa_answer(
                query,
                evidence_rows,
                answer,
                evidence_quality=evidence_quality,
                llm_client=client,
            )
            llm_result.setdefault("llm_metadata", {})
            llm_result["llm_metadata"]["evidence_count"] = len(evidence_rows)
        answer = llm_result["answer"]
        generation_mode = llm_result["generation_mode"]
        llm_metadata = llm_result["llm_metadata"]

    answer_time_ms = elapsed_ms(answer_start)
    return {
        "query": query,
        "answer": answer,
        "evidence_quality": evidence_quality,
        "citations": citation_summary(citation_rows),
        "disclaimer": API_DISCLAIMER,
        "generation_mode": generation_mode,
        "llm_metadata": llm_metadata,
        "timing_metadata": {
            "retrieval_time_ms": retrieval_time_ms,
            "answer_time_ms": answer_time_ms,
            "total_time_ms": elapsed_ms(total_start),
        },
    }


def build_workflow_response(query: str, top_k: int = 5, retrieved_results: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    total_start = start_timer()
    retrieval_start = start_timer()
    results = retrieved_results if retrieved_results is not None else _run_retrieval(query, top_k)
    retrieval_time_ms = elapsed_ms(retrieval_start)

    workflow_start = start_timer()
    context = prepare_workflow_context(query, results)
    triage = context["triage"]
    grounded_answer = render_grounded_answer_section(context["direct_answer"], context["explanation"], context["evidence_lines"])
    workflow_time_ms = elapsed_ms(workflow_start)
    return {
        "query": query,
        "grounded_answer": grounded_answer,
        "risk_level": triage.risk_level,
        "risk_category": triage.risk_category,
        "risk_reasoning": triage.reasoning,
        "workflow_checklist": context["checklist"],
        "citations": citation_summary(context["citation_rows"]),
        "evidence_quality": evidence_note(context["assessment"]),
        "disclaimer": API_DISCLAIMER,
        "timing_metadata": {
            "retrieval_time_ms": retrieval_time_ms,
            "workflow_time_ms": workflow_time_ms,
            "total_time_ms": elapsed_ms(total_start),
        },
    }


def create_case_from_workflow(query: str, requester: str, department: str, top_k: int = 5) -> dict[str, Any]:
    advice = build_workflow_response(query, top_k=top_k)
    record = {
        "query": query,
        "requester": requester,
        "department": department,
        "risk_level": advice["risk_level"],
        "risk_category": advice["risk_category"],
        "recommended_next_steps": advice["workflow_checklist"],
        "citation_summary": advice["citations"],
        "evidence_quality": advice["evidence_quality"],
        "disclaimer": advice["disclaimer"],
    }
    return mock_case_store.create_case(record)


def list_cases() -> list[dict[str, Any]]:
    return mock_case_store.list_cases()


def get_case(case_id: str) -> dict[str, Any] | None:
    return mock_case_store.get_case(case_id)


def start_agent_run_service(
    query: str,
    requester: str = "Demo User",
    department: str = "Compliance",
    top_k: int = 5,
) -> dict[str, Any]:
    total_start = start_timer()
    agent_start = start_timer()
    state = start_agent_run(query, requester=requester, department=department, top_k=top_k)
    agent_run_time_ms = elapsed_ms(agent_start)

    store_start = start_timer()
    response = agent_run_store.create_run(state)
    store_time_ms = elapsed_ms(store_start)

    audit_start = start_timer()
    _log_agent_start_events(response, actor=state.requester)
    audit_time_ms = elapsed_ms(audit_start)
    response["timing_metadata"] = {
        "agent_run_time_ms": agent_run_time_ms,
        "store_time_ms": store_time_ms,
        "audit_time_ms": audit_time_ms,
        "total_time_ms": elapsed_ms(total_start),
    }
    return response


def get_agent_run_service(run_id: str) -> dict[str, Any] | None:
    return agent_run_store.get_run(run_id)


def list_agent_runs_service() -> list[dict[str, Any]]:
    return agent_run_store.list_runs()


def approve_agent_run_service(run_id: str, approver: str = "Human Reviewer") -> dict[str, Any] | None:
    state = agent_run_store.get_run_state(run_id)
    if state is None:
        return None
    was_blocked = state.decision.blocked
    was_case_created = state.case_created
    updated = approve_agent_action(state, approver=approver)
    response = agent_run_store.update_run(run_id, updated)
    if response is None:
        return None

    risk_level, risk_category = _agent_risk_fields(response)
    metadata = _agent_audit_metadata(response)
    if was_blocked or response.get("blocked"):
        _safe_create_audit_event(
            audit_log_store.APPROVAL_BLOCKED,
            "Approval could not create a case because the agent run is blocked.",
            actor=approver,
            run_id=run_id,
            risk_level=risk_level,
            risk_category=risk_category,
            metadata=metadata,
        )
        _safe_create_audit_event(
            audit_log_store.CASE_CREATION_BLOCKED,
            "Mock case creation was blocked by the approval gate.",
            actor=approver,
            run_id=run_id,
            risk_level=risk_level,
            risk_category=risk_category,
            metadata=metadata,
        )
    elif not was_case_created:
        _safe_create_audit_event(
            audit_log_store.APPROVAL_GRANTED,
            "Human approval granted for agent run.",
            actor=approver,
            run_id=run_id,
            risk_level=risk_level,
            risk_category=risk_category,
            metadata=metadata,
        )
        if response.get("case_created") and response.get("case_id"):
            _safe_create_audit_event(
                audit_log_store.CASE_CREATED,
                "Mock compliance case created from approved agent run.",
                actor=approver,
                run_id=run_id,
                case_id=response.get("case_id"),
                risk_level=risk_level,
                risk_category=risk_category,
                metadata=metadata,
            )
    return response


def reject_agent_run_service(run_id: str, reviewer: str = "Human Reviewer", reason: str | None = None) -> dict[str, Any] | None:
    state = agent_run_store.get_run_state(run_id)
    if state is None:
        return None
    updated = reject_agent_action(state, reviewer=reviewer, reason=reason)
    response = agent_run_store.update_run(run_id, updated)
    if response is None:
        return None
    risk_level, risk_category = _agent_risk_fields(response)
    metadata = dict(_agent_audit_metadata(response), reason=reason)
    _safe_create_audit_event(
        audit_log_store.APPROVAL_REJECTED,
        "Human reviewer rejected the pending agent action.",
        actor=reviewer,
        run_id=run_id,
        risk_level=risk_level,
        risk_category=risk_category,
        metadata=metadata,
    )
    return response


def list_audit_logs_service(
    event_type: str | None = None,
    run_id: str | None = None,
    case_id: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    return audit_log_store.list_audit_events(limit=limit, event_type=event_type, run_id=run_id, case_id=case_id)


def get_audit_log_service(audit_id: str) -> dict[str, Any] | None:
    return audit_log_store.get_audit_event(audit_id)
