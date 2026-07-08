"""HTTP client helpers for the local FastAPI backend."""

from __future__ import annotations

from typing import Any

import requests


DEFAULT_TIMEOUT_SECONDS = 30
BACKEND_UNREACHABLE_MESSAGE = (
    "FastAPI backend is not reachable. Start it with: "
    "python3 -m uvicorn backend.main:app --reload --port 8000"
)


def result(ok: bool, data: Any = None, error: str | None = None) -> dict[str, Any]:
    return {"ok": ok, "data": data, "error": error}


def normalize_base_url(base_url: str) -> str:
    return (base_url or "").strip().rstrip("/")


def request_json(method: str, base_url: str, path: str, payload: dict[str, Any] | None = None, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> dict[str, Any]:
    base = normalize_base_url(base_url)
    if not base:
        return result(False, error="API base URL cannot be blank.")
    url = f"{base}{path}"
    try:
        response = requests.request(method, url, json=payload, timeout=timeout)
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return result(False, error=BACKEND_UNREACHABLE_MESSAGE)
    except requests.exceptions.RequestException as exc:
        return result(False, error=f"API request failed: {exc}")

    if response.status_code < 200 or response.status_code >= 300:
        detail = None
        try:
            detail = response.json().get("detail")
        except ValueError:
            detail = response.text.strip() or response.reason
        return result(False, error=f"API returned HTTP {response.status_code}: {detail}")

    try:
        return result(True, data=response.json())
    except ValueError:
        return result(False, error="API returned a non-JSON response.")


def request_json_with_params(method: str, base_url: str, path: str, params: dict[str, Any] | None = None, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> dict[str, Any]:
    base = normalize_base_url(base_url)
    if not base:
        return result(False, error="API base URL cannot be blank.")
    url = f"{base}{path}"
    try:
        response = requests.request(method, url, params=params, timeout=timeout)
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return result(False, error=BACKEND_UNREACHABLE_MESSAGE)
    except requests.exceptions.RequestException as exc:
        return result(False, error=f"API request failed: {exc}")

    if response.status_code < 200 or response.status_code >= 300:
        detail = None
        try:
            detail = response.json().get("detail")
        except ValueError:
            detail = response.text.strip() or response.reason
        return result(False, error=f"API returned HTTP {response.status_code}: {detail}")

    try:
        return result(True, data=response.json())
    except ValueError:
        return result(False, error="API returned a non-JSON response.")


def require_fields(api_result: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    if not api_result.get("ok"):
        return api_result
    data = api_result.get("data")
    if not isinstance(data, dict):
        return result(False, error="API returned an unexpected response shape.")
    missing = [field for field in fields if field not in data]
    if missing:
        return result(False, error="API response is missing expected field(s): " + ", ".join(missing))
    return api_result


def check_health(base_url: str) -> dict[str, Any]:
    return require_fields(request_json("GET", base_url, "/health"), ["status", "service", "mode"])


def ask_qa(base_url: str, query: str, top_k: int = 5, use_llm: bool = False) -> dict[str, Any]:
    payload = {"query": query, "top_k": top_k, "use_llm": use_llm}
    return require_fields(
        request_json("POST", base_url, "/qa", payload),
        ["query", "answer", "evidence_quality", "citations", "disclaimer", "generation_mode", "llm_metadata"],
    )


def get_workflow_advice(base_url: str, query: str, top_k: int = 5) -> dict[str, Any]:
    payload = {"query": query, "top_k": top_k}
    return require_fields(
        request_json("POST", base_url, "/workflow-advice", payload),
        ["query", "grounded_answer", "risk_level", "risk_category", "risk_reasoning", "workflow_checklist", "citations", "evidence_quality", "disclaimer"],
    )


def create_case(base_url: str, query: str, requester: str, department: str, top_k: int = 5) -> dict[str, Any]:
    payload = {"query": query, "requester": requester, "department": department, "top_k": top_k}
    return require_fields(
        request_json("POST", base_url, "/cases", payload),
        ["case_id", "query", "requester", "department", "risk_level", "risk_category", "status", "recommended_next_steps", "created_at", "citation_summary", "evidence_quality", "disclaimer"],
    )


def list_cases(base_url: str) -> dict[str, Any]:
    return require_fields(request_json("GET", base_url, "/cases"), ["cases"])


def get_case(base_url: str, case_id: str) -> dict[str, Any]:
    case_id = (case_id or "").strip()
    if not case_id:
        return result(False, error="Case ID cannot be blank.")
    return require_fields(
        request_json("GET", base_url, f"/cases/{case_id}"),
        ["case_id", "query", "requester", "department", "risk_level", "risk_category", "status", "recommended_next_steps", "created_at", "citation_summary", "evidence_quality", "disclaimer"],
    )


AGENT_RUN_RESPONSE_FIELDS = ["run_id", "status", "agent_state", "approval_required", "blocked", "pending_action", "trace", "case_created", "case_id"]


def start_agent_run(base_url: str, query: str, requester: str = "Demo User", department: str = "Compliance", top_k: int = 5) -> dict[str, Any]:
    payload = {"query": query, "requester": requester, "department": department, "top_k": top_k}
    return require_fields(request_json("POST", base_url, "/agent-runs", payload), AGENT_RUN_RESPONSE_FIELDS)


def list_agent_runs(base_url: str) -> dict[str, Any]:
    return require_fields(request_json("GET", base_url, "/agent-runs"), ["agent_runs"])


def get_agent_run(base_url: str, run_id: str) -> dict[str, Any]:
    run_id = (run_id or "").strip()
    if not run_id:
        return result(False, error="Agent run ID cannot be blank.")
    return require_fields(request_json("GET", base_url, f"/agent-runs/{run_id}"), AGENT_RUN_RESPONSE_FIELDS)


def approve_agent_run(base_url: str, run_id: str, approver: str = "Human Reviewer") -> dict[str, Any]:
    run_id = (run_id or "").strip()
    if not run_id:
        return result(False, error="Agent run ID cannot be blank.")
    payload = {"approver": approver}
    return require_fields(request_json("POST", base_url, f"/agent-runs/{run_id}/approve", payload), AGENT_RUN_RESPONSE_FIELDS)


def reject_agent_run(base_url: str, run_id: str, reviewer: str = "Human Reviewer", reason: str | None = None) -> dict[str, Any]:
    run_id = (run_id or "").strip()
    if not run_id:
        return result(False, error="Agent run ID cannot be blank.")
    payload = {"reviewer": reviewer, "reason": reason}
    return require_fields(request_json("POST", base_url, f"/agent-runs/{run_id}/reject", payload), AGENT_RUN_RESPONSE_FIELDS)


AUDIT_EVENT_RESPONSE_FIELDS = ["audit_id", "timestamp", "event_type", "message", "metadata"]


def list_audit_logs(
    base_url: str,
    event_type: str | None = None,
    run_id: str | None = None,
    case_id: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    params: dict[str, Any] = {"limit": limit}
    if event_type and event_type.strip():
        params["event_type"] = event_type.strip()
    if run_id and run_id.strip():
        params["run_id"] = run_id.strip()
    if case_id and case_id.strip():
        params["case_id"] = case_id.strip()
    return require_fields(request_json_with_params("GET", base_url, "/audit-logs", params=params), ["audit_events"])


def get_audit_log(base_url: str, audit_id: str) -> dict[str, Any]:
    audit_id = (audit_id or "").strip()
    if not audit_id:
        return result(False, error="Audit event ID cannot be blank.")
    return require_fields(request_json("GET", base_url, f"/audit-logs/{audit_id}"), AUDIT_EVENT_RESPONSE_FIELDS)
