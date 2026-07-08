from fastapi.testclient import TestClient

from backend import agent_run_store, audit_log_store, mock_case_store, services
from backend.main import app
from src.agent.human_loop_agent import start_agent_run as build_agent_state


client = TestClient(app)


def sample_workflow_response(query="query", risk_level="High", risk_category="Insider Trading / Material Non-Public Information"):
    return {
        "query": query,
        "grounded_answer": "FULL GROUNDED ANSWER TEXT SHOULD NOT BE AUDITED.",
        "risk_level": risk_level,
        "risk_category": risk_category,
        "risk_reasoning": "The query and evidence match the risk category. [C1]",
        "workflow_checklist": ["Review the issue. [C1]", "Escalate to compliance."],
        "citations": [
            {
                "citation_id": "[C1]",
                "policy_name": "防範內線交易管理程序",
                "article": "第五條",
                "article_title": "重大消息",
                "pages": "1-2",
                "source_file": "policy.pdf",
                "chunk_id": "chunk-1",
                "text": "FULL EVIDENCE TEXT SHOULD NOT BE AUDITED.",
            }
        ],
        "evidence_quality": "Evidence quality appears sufficient.",
        "disclaimer": "Local demo only.",
    }


def clear_stores():
    audit_log_store.clear_audit_events()
    agent_run_store.clear_runs()
    mock_case_store.clear_cases()


def setup_function():
    clear_stores()


def mock_start_agent_run(monkeypatch, risk_level="High", risk_category="Insider Trading / Material Non-Public Information"):
    def fake_start_agent_run(query, requester="Demo User", department="Compliance", top_k=5):
        return build_agent_state(
            query,
            requester=requester,
            department=department,
            top_k=top_k,
            workflow_func=lambda workflow_query, workflow_top_k: sample_workflow_response(workflow_query, risk_level, risk_category),
        )

    monkeypatch.setattr(services, "start_agent_run", fake_start_agent_run)


def event_types():
    return [event["event_type"] for event in audit_log_store.list_audit_events()]


def test_starting_high_risk_agent_run_writes_created_and_approval_required(monkeypatch):
    mock_start_agent_run(monkeypatch, "High")

    response = client.post("/agent-runs", json={"query": "query", "requester": "Demo User", "department": "Compliance", "top_k": 5})

    assert response.status_code == 200
    assert event_types() == [audit_log_store.AGENT_RUN_CREATED, audit_log_store.APPROVAL_REQUIRED]
    assert audit_log_store.list_audit_events()[0]["run_id"] == response.json()["run_id"]


def test_starting_insufficient_evidence_run_writes_blocked_event(monkeypatch):
    mock_start_agent_run(monkeypatch, "Insufficient Evidence", "Unknown or Insufficient Evidence")

    response = client.post("/agent-runs", json={"query": "query"})

    assert response.status_code == 200
    assert response.json()["blocked"] is True
    assert event_types() == [audit_log_store.AGENT_RUN_CREATED, audit_log_store.CASE_BLOCKED_INSUFFICIENT_EVIDENCE]


def test_approving_high_risk_pending_run_writes_approval_and_case_created(monkeypatch):
    mock_start_agent_run(monkeypatch, "High")
    created = client.post("/agent-runs", json={"query": "query"}).json()

    response = client.post(f"/agent-runs/{created['run_id']}/approve", json={"approver": "Reviewer"})
    payload = response.json()

    assert response.status_code == 200
    assert payload["case_created"] is True
    assert audit_log_store.APPROVAL_GRANTED in event_types()
    assert audit_log_store.CASE_CREATED in event_types()
    case_events = audit_log_store.list_audit_events(event_type=audit_log_store.CASE_CREATED)
    assert case_events[0]["case_id"] == payload["case_id"]


def test_rejecting_high_risk_pending_run_writes_rejection_and_no_case_created(monkeypatch):
    mock_start_agent_run(monkeypatch, "High")
    created = client.post("/agent-runs", json={"query": "query"}).json()

    response = client.post(f"/agent-runs/{created['run_id']}/reject", json={"reviewer": "Reviewer", "reason": "Need context."})

    assert response.status_code == 200
    assert audit_log_store.APPROVAL_REJECTED in event_types()
    assert audit_log_store.CASE_CREATED not in event_types()
    rejection = audit_log_store.list_audit_events(event_type=audit_log_store.APPROVAL_REJECTED)[0]
    assert rejection["metadata"]["reason"] == "Need context."


def test_approving_blocked_run_writes_blocked_events_and_no_case_created(monkeypatch):
    mock_start_agent_run(monkeypatch, "Insufficient Evidence", "Unknown or Insufficient Evidence")
    created = client.post("/agent-runs", json={"query": "query"}).json()

    response = client.post(f"/agent-runs/{created['run_id']}/approve", json={"approver": "Reviewer"})

    assert response.status_code == 200
    assert response.json()["case_created"] is False
    assert audit_log_store.APPROVAL_BLOCKED in event_types()
    assert audit_log_store.CASE_CREATION_BLOCKED in event_types()
    assert audit_log_store.CASE_CREATED not in event_types()


def test_get_audit_logs_returns_events_and_supports_filters(monkeypatch):
    mock_start_agent_run(monkeypatch, "High")
    created = client.post("/agent-runs", json={"query": "query"}).json()
    approved = client.post(f"/agent-runs/{created['run_id']}/approve", json={"approver": "Reviewer"}).json()

    all_events = client.get("/audit-logs")
    approval_required = client.get("/audit-logs", params={"event_type": audit_log_store.APPROVAL_REQUIRED})
    by_run = client.get("/audit-logs", params={"run_id": created["run_id"]})
    by_case = client.get("/audit-logs", params={"case_id": approved["case_id"]})

    assert all_events.status_code == 200
    assert len(all_events.json()["audit_events"]) >= 4
    assert [event["event_type"] for event in approval_required.json()["audit_events"]] == [audit_log_store.APPROVAL_REQUIRED]
    assert all(event["run_id"] == created["run_id"] for event in by_run.json()["audit_events"])
    assert [event["event_type"] for event in by_case.json()["audit_events"]] == [audit_log_store.CASE_CREATED]


def test_get_audit_log_returns_one_event_and_missing_returns_404(monkeypatch):
    mock_start_agent_run(monkeypatch, "High")
    client.post("/agent-runs", json={"query": "query"})
    audit_id = audit_log_store.list_audit_events()[0]["audit_id"]

    response = client.get(f"/audit-logs/{audit_id}")
    missing = client.get("/audit-logs/AUDIT-MISSING")

    assert response.status_code == 200
    assert response.json()["audit_id"] == audit_id
    assert missing.status_code == 404


def test_audit_events_do_not_include_full_answers_or_evidence_text(monkeypatch):
    mock_start_agent_run(monkeypatch, "High")
    client.post("/agent-runs", json={"query": "query"})

    serialized = str(audit_log_store.list_audit_events())

    assert "FULL GROUNDED ANSWER TEXT SHOULD NOT BE AUDITED" not in serialized
    assert "FULL EVIDENCE TEXT SHOULD NOT BE AUDITED" not in serialized
    assert "workflow_response" not in serialized
    assert "citations" not in serialized
