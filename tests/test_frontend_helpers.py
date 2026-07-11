import requests

from frontend import api_client
from frontend.streamlit_app import (
    display_article_title,
    display_source_file,
    format_generation_mode,
    format_llm_status_message,
    format_timing_metadata,
    prepare_citations_for_display,
    safe_llm_metadata,
)
from frontend.evaluation_examples import evaluate_response, summarize_results


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text="", reason="OK", json_error=False):
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self.reason = reason
        self._json_error = json_error

    def json(self):
        if self._json_error:
            raise ValueError("not json")
        return self._payload


def test_format_timing_metadata_formats_known_keys_in_stable_order():
    formatted = format_timing_metadata(
        {
            "total_time_ms": 205.9,
            "workflow_time_ms": 120.444,
            "retrieval_time_ms": 82.31,
        }
    )

    assert formatted == "retrieval: 82.31 ms · workflow: 120.44 ms · total: 205.90 ms"


def test_format_timing_metadata_returns_empty_for_missing_values():
    assert format_timing_metadata(None) == ""
    assert format_timing_metadata({}) == ""


def test_format_timing_metadata_includes_unknown_numeric_fields_after_known_keys():
    formatted = format_timing_metadata(
        {
            "store_time_ms": 1.234,
            "total_time_ms": 10,
            "custom_latency_ms": 2.5,
        }
    )

    assert formatted == "total: 10.00 ms · store time: 1.23 ms · custom latency: 2.50 ms"


def test_format_timing_metadata_ignores_nested_dict_values():
    formatted = format_timing_metadata(
        {
            "retrieval_time_ms": 3.456,
            "nested": {"total_time_ms": 10},
        }
    )

    assert formatted == "retrieval: 3.46 ms"


def test_format_timing_metadata_skips_non_numeric_values_without_crashing():
    formatted = format_timing_metadata(
        {
            "retrieval_time_ms": "fast",
            "answer_time_ms": None,
            "total_time_ms": 20.0,
        }
    )

    assert formatted == "total: 20.00 ms"


def test_format_timing_metadata_does_not_treat_bool_as_numeric():
    formatted = format_timing_metadata(
        {
            "retrieval_time_ms": True,
            "total_time_ms": False,
            "workflow_time_ms": 8,
        }
    )

    assert formatted == "workflow: 8.00 ms"


def test_evaluate_response_accepts_multiple_expected_values():
    example = {
        "query": "關係人交易需要董事會核准嗎？",
        "expected_risks": ["Medium", "High"],
        "expected_categories": ["Related-Party Transaction", "Board Approval / Governance Procedure"],
    }
    response = {
        "risk_level": "High",
        "risk_category": "Board Approval / Governance Procedure",
        "evidence_quality": "ok",
    }

    row = evaluate_response(example, response)

    assert row["pass"] is True
    assert row["expected_risk"] == "Medium or High"


def test_evaluate_response_fails_when_risk_or_category_mismatch():
    example = {
        "query": "員工旅遊補助",
        "expected_risks": ["Insufficient Evidence"],
        "expected_categories": ["Unknown or Insufficient Evidence"],
    }
    response = {"risk_level": "Low", "risk_category": "Ethical Conduct / Conflict of Interest"}

    row = evaluate_response(example, response)

    assert row["pass"] is False


def test_summarize_results_computes_pass_rate():
    summary = summarize_results([{"pass": True}, {"pass": False}, {"pass": True}])

    assert summary["total_examples"] == 3
    assert summary["passed_examples"] == 2
    assert summary["pass_rate"] == 2 / 3


def test_api_client_success_path_and_required_fields(monkeypatch):
    def fake_request(method, url, json=None, timeout=30):
        return FakeResponse(payload={"status": "ok", "service": "svc", "mode": "local-demo"})

    monkeypatch.setattr(api_client.requests, "request", fake_request)

    result = api_client.check_health("http://127.0.0.1:8000")

    assert result["ok"] is True
    assert result["data"]["status"] == "ok"


def test_api_client_connection_error_is_user_friendly(monkeypatch):
    def fake_request(method, url, json=None, timeout=30):
        raise requests.exceptions.ConnectionError("boom")

    monkeypatch.setattr(api_client.requests, "request", fake_request)

    result = api_client.check_health("http://127.0.0.1:8000")

    assert result["ok"] is False
    assert "FastAPI backend is not reachable" in result["error"]


def test_api_client_timeout_is_user_friendly(monkeypatch):
    def fake_request(method, url, json=None, timeout=30):
        raise requests.exceptions.Timeout("slow")

    monkeypatch.setattr(api_client.requests, "request", fake_request)

    result = api_client.ask_qa("http://127.0.0.1:8000", "query", 5)

    assert result["ok"] is False
    assert "Start it with" in result["error"]


def qa_payload():
    return {
        "query": "query",
        "answer": "answer",
        "evidence_quality": "Sufficient",
        "citations": [],
        "disclaimer": "demo",
        "generation_mode": "deterministic",
        "llm_metadata": {"provider": "deepseek", "used_llm": False},
    }


def test_ask_qa_default_sends_use_llm_false(monkeypatch):
    calls = []

    def fake_request(method, url, json=None, timeout=30):
        calls.append({"method": method, "url": url, "json": json})
        return FakeResponse(payload=qa_payload())

    monkeypatch.setattr(api_client.requests, "request", fake_request)

    result = api_client.ask_qa("http://127.0.0.1:8000", "query", 5)

    assert result["ok"] is True
    assert calls == [
        {
            "method": "POST",
            "url": "http://127.0.0.1:8000/qa",
            "json": {"query": "query", "top_k": 5, "use_llm": False},
        }
    ]


def test_ask_qa_sends_use_llm_true(monkeypatch):
    calls = []

    def fake_request(method, url, json=None, timeout=30):
        calls.append({"method": method, "url": url, "json": json})
        return FakeResponse(payload=dict(qa_payload(), generation_mode="llm_grounded", llm_metadata={"provider": "deepseek", "used_llm": True}))

    monkeypatch.setattr(api_client.requests, "request", fake_request)

    result = api_client.ask_qa("http://127.0.0.1:8000", "query", 5, use_llm=True)

    assert result["ok"] is True
    assert calls[0]["json"]["use_llm"] is True
    assert result["data"]["generation_mode"] == "llm_grounded"


def test_api_client_non_200_response_returns_error(monkeypatch):
    def fake_request(method, url, json=None, timeout=30):
        return FakeResponse(status_code=503, payload={"detail": "service unavailable"})

    monkeypatch.setattr(api_client.requests, "request", fake_request)

    result = api_client.ask_qa("http://127.0.0.1:8000", "query", 5)

    assert result["ok"] is False
    assert "HTTP 503" in result["error"]
    assert "service unavailable" in result["error"]


def test_api_client_non_json_response_returns_error(monkeypatch):
    def fake_request(method, url, json=None, timeout=30):
        return FakeResponse(status_code=200, text="html", json_error=True)

    monkeypatch.setattr(api_client.requests, "request", fake_request)

    result = api_client.check_health("http://127.0.0.1:8000")

    assert result["ok"] is False
    assert "non-JSON" in result["error"]


def test_api_client_missing_expected_fields_returns_error(monkeypatch):
    def fake_request(method, url, json=None, timeout=30):
        return FakeResponse(payload={"answer": "ok"})

    monkeypatch.setattr(api_client.requests, "request", fake_request)

    result = api_client.ask_qa("http://127.0.0.1:8000", "query", 5)

    assert result["ok"] is False
    assert "missing expected field" in result["error"]


def test_format_generation_mode_defaults_to_deterministic():
    assert format_generation_mode(None) == "deterministic"
    assert format_generation_mode("  ") == "deterministic"
    assert format_generation_mode("llm_grounded") == "llm_grounded"


def test_safe_llm_metadata_keeps_only_safe_fields():
    metadata = {
        "provider": "deepseek",
        "used_llm": True,
        "fallback_reason": "missing_citation_reference",
        "evidence_count": 2,
        "configured": True,
        "prompt": "hidden prompt",
        "api_key": "secret",
        "Authorization": "Bearer secret",
        "raw_request_payload": {"x": "y"},
    }

    safe = safe_llm_metadata(metadata)

    assert safe == {
        "provider": "deepseek",
        "used_llm": True,
        "fallback_reason": "missing_citation_reference",
        "evidence_count": 2,
        "configured": True,
    }
    assert "prompt" not in safe
    assert "api_key" not in safe
    assert "Authorization" not in safe


def test_format_llm_status_message_describes_grounded_and_fallback_modes():
    grounded = format_llm_status_message("llm_grounded", {"provider": "deepseek", "used_llm": True})
    fallback = format_llm_status_message("llm_fallback", {"fallback_reason": "LLMConfigurationError"})
    deterministic = format_llm_status_message("deterministic", None)

    assert "retrieved evidence" in grounded
    assert "LLMConfigurationError" in fallback
    assert deterministic == "Generation mode: deterministic."


def test_get_case_rejects_blank_case_id_without_api_call():
    result = api_client.get_case("http://127.0.0.1:8000", "   ")

    assert result["ok"] is False
    assert "Case ID cannot be blank" in result["error"]


def agent_run_payload():
    return {
        "run_id": "AGENT-RUN-0001",
        "status": "PendingApproval",
        "agent_state": {
            "risk_level": "High",
            "risk_category": "Insider Trading / Material Non-Public Information",
            "workflow_response": {},
        },
        "approval_required": True,
        "blocked": False,
        "pending_action": {"action_type": "create_mock_case"},
        "trace": [{"step": "intake", "status": "received", "message": "ok"}],
        "case_created": False,
        "case_id": None,
    }


def test_start_agent_run_posts_expected_payload(monkeypatch):
    calls = []

    def fake_request(method, url, json=None, timeout=30):
        calls.append({"method": method, "url": url, "json": json})
        return FakeResponse(payload=agent_run_payload())

    monkeypatch.setattr(api_client.requests, "request", fake_request)

    result = api_client.start_agent_run("http://127.0.0.1:8000", "query", requester="Demo User", department="Compliance", top_k=5)

    assert result["ok"] is True
    assert calls == [
        {
            "method": "POST",
            "url": "http://127.0.0.1:8000/agent-runs",
            "json": {"query": "query", "requester": "Demo User", "department": "Compliance", "top_k": 5},
        }
    ]


def test_start_agent_run_validates_required_response_fields(monkeypatch):
    def fake_request(method, url, json=None, timeout=30):
        return FakeResponse(payload={"run_id": "AGENT-RUN-0001"})

    monkeypatch.setattr(api_client.requests, "request", fake_request)

    result = api_client.start_agent_run("http://127.0.0.1:8000", "query")

    assert result["ok"] is False
    assert "missing expected field" in result["error"]
    assert "agent_state" in result["error"]


def test_list_agent_runs_validates_response(monkeypatch):
    def fake_request(method, url, json=None, timeout=30):
        return FakeResponse(payload={"agent_runs": [{"run_id": "AGENT-RUN-0001"}]})

    monkeypatch.setattr(api_client.requests, "request", fake_request)

    result = api_client.list_agent_runs("http://127.0.0.1:8000")

    assert result["ok"] is True
    assert result["data"]["agent_runs"][0]["run_id"] == "AGENT-RUN-0001"


def test_get_agent_run_rejects_blank_run_id_without_api_call(monkeypatch):
    def fake_request(method, url, json=None, timeout=30):
        raise AssertionError("request should not be called")

    monkeypatch.setattr(api_client.requests, "request", fake_request)

    result = api_client.get_agent_run("http://127.0.0.1:8000", "   ")

    assert result["ok"] is False
    assert "Agent run ID cannot be blank" in result["error"]


def test_approve_agent_run_posts_to_approve_endpoint(monkeypatch):
    calls = []

    def fake_request(method, url, json=None, timeout=30):
        calls.append({"method": method, "url": url, "json": json})
        return FakeResponse(payload=dict(agent_run_payload(), status="CaseCreated", case_created=True, case_id="CASE-1"))

    monkeypatch.setattr(api_client.requests, "request", fake_request)

    result = api_client.approve_agent_run("http://127.0.0.1:8000", "AGENT-RUN-0001", approver="Reviewer")

    assert result["ok"] is True
    assert calls == [
        {
            "method": "POST",
            "url": "http://127.0.0.1:8000/agent-runs/AGENT-RUN-0001/approve",
            "json": {"approver": "Reviewer"},
        }
    ]


def test_reject_agent_run_posts_reason_to_reject_endpoint(monkeypatch):
    calls = []

    def fake_request(method, url, json=None, timeout=30):
        calls.append({"method": method, "url": url, "json": json})
        return FakeResponse(payload=dict(agent_run_payload(), status="Rejected"))

    monkeypatch.setattr(api_client.requests, "request", fake_request)

    result = api_client.reject_agent_run("http://127.0.0.1:8000", "AGENT-RUN-0001", reviewer="Reviewer", reason="Need context.")

    assert result["ok"] is True
    assert calls == [
        {
            "method": "POST",
            "url": "http://127.0.0.1:8000/agent-runs/AGENT-RUN-0001/reject",
            "json": {"reviewer": "Reviewer", "reason": "Need context."},
        }
    ]


def audit_event_payload():
    return {
        "audit_id": "AUDIT-0001",
        "timestamp": "2026-07-08T00:00:00+00:00",
        "event_type": "AGENT_RUN_CREATED",
        "actor": "Demo User",
        "run_id": "AGENT-RUN-0001",
        "case_id": None,
        "risk_level": "High",
        "risk_category": "Insider Trading / Material Non-Public Information",
        "message": "Agent run created.",
        "metadata": {"status": "PendingApproval"},
    }


def test_list_audit_logs_calls_endpoint_with_query_params(monkeypatch):
    calls = []

    def fake_request(method, url, json=None, params=None, timeout=30):
        calls.append({"method": method, "url": url, "params": params})
        return FakeResponse(payload={"audit_events": [audit_event_payload()]})

    monkeypatch.setattr(api_client.requests, "request", fake_request)

    result = api_client.list_audit_logs(
        "http://127.0.0.1:8000",
        event_type="AGENT_RUN_CREATED",
        run_id="AGENT-RUN-0001",
        case_id="CASE-1",
        limit=10,
    )

    assert result["ok"] is True
    assert calls == [
        {
            "method": "GET",
            "url": "http://127.0.0.1:8000/audit-logs",
            "params": {
                "event_type": "AGENT_RUN_CREATED",
                "run_id": "AGENT-RUN-0001",
                "case_id": "CASE-1",
                "limit": 10,
            },
        }
    ]


def test_list_audit_logs_omits_empty_string_filters(monkeypatch):
    calls = []

    def fake_request(method, url, json=None, params=None, timeout=30):
        calls.append(params)
        return FakeResponse(payload={"audit_events": []})

    monkeypatch.setattr(api_client.requests, "request", fake_request)

    result = api_client.list_audit_logs("http://127.0.0.1:8000", event_type=" ", run_id="", case_id=" ", limit=20)

    assert result["ok"] is True
    assert calls == [{"limit": 20}]


def test_get_audit_log_rejects_blank_audit_id_without_api_call(monkeypatch):
    def fake_request(method, url, json=None, params=None, timeout=30):
        raise AssertionError("request should not be called")

    monkeypatch.setattr(api_client.requests, "request", fake_request)

    result = api_client.get_audit_log("http://127.0.0.1:8000", "   ")

    assert result["ok"] is False
    assert "Audit event ID cannot be blank" in result["error"]


def test_get_audit_log_calls_detail_endpoint_and_validates_response(monkeypatch):
    calls = []

    def fake_request(method, url, json=None, params=None, timeout=30):
        calls.append({"method": method, "url": url, "json": json, "params": params})
        return FakeResponse(payload=audit_event_payload())

    monkeypatch.setattr(api_client.requests, "request", fake_request)

    result = api_client.get_audit_log("http://127.0.0.1:8000", "AUDIT-0001")

    assert result["ok"] is True
    assert calls == [
        {
            "method": "GET",
            "url": "http://127.0.0.1:8000/audit-logs/AUDIT-0001",
            "json": None,
            "params": None,
        }
    ]


def test_streamlit_citation_display_uses_basename_and_title_fallback():
    rows = prepare_citations_for_display([
        {
            "citation_id": "[C1]",
            "policy_name": "政策",
            "article": "第一條",
            "article_title": "TBD",
            "pages": "1-1",
            "source_file": "/Users/demo/data/raw_pdfs/policy.pdf",
            "chunk_id": "chunk-1",
        }
    ])

    assert rows[0]["source_file"] == "policy.pdf"
    assert rows[0]["article_title"] == "No explicit title"


def test_streamlit_display_helpers_handle_blank_values():
    assert display_source_file("") == ""
    assert display_article_title("") == "No explicit title"
