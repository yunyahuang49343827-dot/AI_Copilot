from fastapi.testclient import TestClient

from backend import agent_run_store, mock_case_store, services
from backend.main import app
from src.agent.human_loop_agent import start_agent_run


client = TestClient(app)


def sample_workflow_response(query="query", risk_level="High", risk_category="Insider Trading / Material Non-Public Information"):
    return {
        "query": query,
        "grounded_answer": "Grounded answer omitted from trace.",
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
            }
        ],
        "evidence_quality": "Evidence quality appears sufficient.",
        "disclaimer": "Local demo only.",
    }


def make_agent_run_response(query="query", risk_level="High", risk_category="Insider Trading / Material Non-Public Information"):
    state = start_agent_run(
        query,
        requester="demo_user",
        department="Finance",
        top_k=5,
        workflow_func=lambda workflow_query, top_k: sample_workflow_response(workflow_query, risk_level, risk_category),
    )
    return agent_run_store.create_run(state)


def clear_stores():
    agent_run_store.clear_runs()
    mock_case_store.clear_cases()


def test_post_agent_runs_creates_high_risk_run(monkeypatch):
    clear_stores()
    monkeypatch.setattr(services, "start_agent_run_service", lambda query, requester="Demo User", department="Compliance", top_k=5: make_agent_run_response(query, "High"))

    response = client.post(
        "/agent-runs",
        json={"query": "公司還沒公告重大資訊，可以買股票嗎？", "requester": "demo_user", "department": "Finance", "top_k": 5},
    )
    payload = response.json()

    assert response.status_code == 200
    assert payload["run_id"] == "AGENT-RUN-0001"
    assert payload["approval_required"] is True
    assert payload["blocked"] is False
    assert payload["case_created"] is False
    assert payload["status"] == "PendingApproval"


def test_post_agent_runs_creates_blocked_insufficient_evidence_run(monkeypatch):
    clear_stores()
    monkeypatch.setattr(
        services,
        "start_agent_run_service",
        lambda query, requester="Demo User", department="Compliance", top_k=5: make_agent_run_response(
            query,
            "Insufficient Evidence",
            "Unknown or Insufficient Evidence",
        ),
    )

    response = client.post("/agent-runs", json={"query": "員工旅遊補助", "top_k": 5})
    payload = response.json()

    assert response.status_code == 200
    assert payload["blocked"] is True
    assert payload["status"] == "Blocked"
    assert payload["case_created"] is False


def test_get_agent_runs_lists_created_runs(monkeypatch):
    clear_stores()
    monkeypatch.setattr(services, "start_agent_run_service", lambda query, requester="Demo User", department="Compliance", top_k=5: make_agent_run_response(query, "High"))

    client.post("/agent-runs", json={"query": "query one"})
    client.post("/agent-runs", json={"query": "query two"})

    response = client.get("/agent-runs")
    payload = response.json()

    assert response.status_code == 200
    assert len(payload["agent_runs"]) >= 2
    assert payload["agent_runs"][0]["run_id"] == "AGENT-RUN-0001"
    assert "agent_state" not in payload["agent_runs"][0]


def test_get_agent_run_returns_full_run_and_missing_returns_404(monkeypatch):
    clear_stores()
    monkeypatch.setattr(services, "start_agent_run_service", lambda query, requester="Demo User", department="Compliance", top_k=5: make_agent_run_response(query, "High"))

    created = client.post("/agent-runs", json={"query": "query"}).json()
    response = client.get(f"/agent-runs/{created['run_id']}")
    missing = client.get("/agent-runs/AGENT-RUN-MISSING")

    assert response.status_code == 200
    assert response.json()["agent_state"]["query"] == "query"
    assert missing.status_code == 404
    assert "Agent run not found" in missing.json()["detail"]


def test_approve_agent_run_creates_mock_case_and_is_idempotent(monkeypatch):
    clear_stores()
    monkeypatch.setattr(services, "start_agent_run_service", lambda query, requester="Demo User", department="Compliance", top_k=5: make_agent_run_response(query, "High"))

    created = client.post("/agent-runs", json={"query": "query"}).json()
    approve_response = client.post(f"/agent-runs/{created['run_id']}/approve", json={"approver": "Reviewer"})
    second_response = client.post(f"/agent-runs/{created['run_id']}/approve", json={"approver": "Reviewer"})

    approved = approve_response.json()
    second = second_response.json()
    cases = mock_case_store.list_cases()

    assert approve_response.status_code == 200
    assert approved["case_created"] is True
    assert approved["case_id"] is not None
    assert approved["status"] == "CaseCreated"
    assert second_response.status_code == 200
    assert second["case_id"] == approved["case_id"]
    assert len(cases) == 1


def test_reject_agent_run_records_rejection(monkeypatch):
    clear_stores()
    monkeypatch.setattr(services, "start_agent_run_service", lambda query, requester="Demo User", department="Compliance", top_k=5: make_agent_run_response(query, "High"))

    created = client.post("/agent-runs", json={"query": "query"}).json()
    response = client.post(
        f"/agent-runs/{created['run_id']}/reject",
        json={"reviewer": "Reviewer", "reason": "Need more context."},
    )
    payload = response.json()

    assert response.status_code == 200
    assert payload["case_created"] is False
    assert payload["agent_state"]["human_decision"] == "rejected"
    assert payload["status"] == "Rejected"
    assert any(entry["status"] == "rejected" for entry in payload["trace"])


def test_approving_blocked_insufficient_evidence_run_does_not_create_case(monkeypatch):
    clear_stores()
    monkeypatch.setattr(
        services,
        "start_agent_run_service",
        lambda query, requester="Demo User", department="Compliance", top_k=5: make_agent_run_response(
            query,
            "Insufficient Evidence",
            "Unknown or Insufficient Evidence",
        ),
    )

    created = client.post("/agent-runs", json={"query": "員工旅遊補助"}).json()
    response = client.post(f"/agent-runs/{created['run_id']}/approve", json={"approver": "Reviewer"})
    payload = response.json()

    assert response.status_code == 200
    assert payload["case_created"] is False
    assert payload["status"] == "Blocked"
    assert payload["case_id"] is None
    assert mock_case_store.list_cases() == []


def test_agent_run_missing_approve_and_reject_return_404():
    clear_stores()

    approve = client.post("/agent-runs/AGENT-RUN-MISSING/approve", json={"approver": "Reviewer"})
    reject = client.post("/agent-runs/AGENT-RUN-MISSING/reject", json={"reviewer": "Reviewer"})

    assert approve.status_code == 404
    assert reject.status_code == 404


def test_agent_run_service_error_returns_503(monkeypatch):
    clear_stores()

    def fail(query, requester="Demo User", department="Compliance", top_k=5):
        raise services.RetrievalServiceError("Local retrieval service is unavailable.")

    monkeypatch.setattr(services, "start_agent_run_service", fail)

    response = client.post("/agent-runs", json={"query": "query"})

    assert response.status_code == 503
    assert "Local retrieval service is unavailable" in response.json()["detail"]


def test_existing_health_endpoint_still_works():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
