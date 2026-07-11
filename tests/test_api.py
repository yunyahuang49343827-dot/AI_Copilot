from fastapi.testclient import TestClient

from backend import mock_case_store, services
from backend.main import app


client = TestClient(app)


def sample_citation():
    return {
        "citation_id": "[C1]",
        "policy_name": "防範內線交易管理程序",
        "article": "第五條",
        "article_title": "重大消息",
        "pages": "1-2",
        "source_file": "policy.pdf",
        "chunk_id": "chunk-1",
    }


def sample_qa_response(query="公司還沒公告重大資訊，可以買股票嗎？"):
    return {
        "query": query,
        "answer": "Direct Answer:\nBased on retrieved evidence, human review is required. [C1]",
        "evidence_quality": "Evidence quality appears sufficient.",
        "citations": [sample_citation()],
        "disclaimer": "This is a portfolio demo. Risk level is a demo triage priority based on retrieved policy evidence, not a legal determination.",
        "generation_mode": "deterministic",
        "llm_metadata": {"provider": "deepseek", "used_llm": False},
    }


def sample_workflow_response(query="公司還沒公告重大資訊，可以買股票嗎？"):
    return {
        "query": query,
        "grounded_answer": "Direct Answer:\nBased on retrieved evidence, human review is required. [C1]",
        "risk_level": "High",
        "risk_category": "Insider Trading / Material Non-Public Information",
        "risk_reasoning": "The query and evidence match insider trading risk. [C1]",
        "workflow_checklist": ["Do not trade. [C1]", "Escalate to human review."],
        "citations": [sample_citation()],
        "evidence_quality": "Evidence quality appears sufficient.",
        "disclaimer": "This is a portfolio demo. Risk level is a demo triage priority based on retrieved policy evidence, not a legal determination.",
    }

def test_health_returns_status_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["mode"] == "local-demo"


def test_qa_returns_answer_citations_quality_and_disclaimer(monkeypatch):
    calls = {}

    def fake_build_qa_response(query, top_k=5, use_llm=False):
        calls["use_llm"] = use_llm
        return sample_qa_response(query)

    monkeypatch.setattr(services, "build_qa_response", fake_build_qa_response)

    response = client.post("/qa", json={"query": "公司還沒公告重大資訊，可以買股票嗎？", "top_k": 5})
    payload = response.json()

    assert response.status_code == 200
    assert payload["answer"]
    assert payload["citations"][0]["citation_id"] == "[C1]"
    assert payload["evidence_quality"]
    assert payload["disclaimer"]
    assert payload["generation_mode"] == "deterministic"
    assert payload["llm_metadata"]["used_llm"] is False
    assert calls["use_llm"] is False


def test_qa_accepts_use_llm_true(monkeypatch):
    calls = {}

    def fake_build_qa_response(query, top_k=5, use_llm=False):
        calls["query"] = query
        calls["top_k"] = top_k
        calls["use_llm"] = use_llm
        response = sample_qa_response(query)
        response["answer"] = "LLM answer grounded in WITS-001."
        response["generation_mode"] = "llm_grounded"
        response["llm_metadata"] = {"provider": "deepseek", "used_llm": True, "evidence_count": 1}
        return response

    monkeypatch.setattr(services, "build_qa_response", fake_build_qa_response)

    response = client.post(
        "/qa",
        json={"query": "公司還沒公告重大資訊，可以買股票嗎？", "top_k": 5, "use_llm": True},
    )
    payload = response.json()

    assert response.status_code == 200
    assert calls == {"query": "公司還沒公告重大資訊，可以買股票嗎？", "top_k": 5, "use_llm": True}
    assert payload["answer"] == "LLM answer grounded in WITS-001."
    assert payload["generation_mode"] == "llm_grounded"
    assert payload["llm_metadata"]["used_llm"] is True


def test_workflow_advice_returns_risk_fields_and_checklist(monkeypatch):
    monkeypatch.setattr(services, "build_workflow_response", lambda query, top_k=5: sample_workflow_response(query))

    response = client.post("/workflow-advice", json={"query": "公司還沒公告重大資訊，可以買股票嗎？", "top_k": 5})
    payload = response.json()

    assert response.status_code == 200
    assert payload["risk_level"] == "High"
    assert payload["risk_category"] == "Insider Trading / Material Non-Public Information"
    assert payload["workflow_checklist"]
    assert payload["citations"][0]["chunk_id"] == "chunk-1"


def test_cases_create_list_and_get(monkeypatch):
    mock_case_store.clear_cases()
    monkeypatch.setattr(services, "build_workflow_response", lambda query, top_k=5: sample_workflow_response(query))

    create_response = client.post(
        "/cases",
        json={"query": "公司還沒公告重大資訊，可以買股票嗎？", "requester": "demo_user", "department": "Finance", "top_k": 5},
    )
    created = create_response.json()

    assert create_response.status_code == 200
    assert created["case_id"].startswith("CASE-")
    assert created["status"] == "Created"
    assert created["citation_summary"][0]["policy_name"] == "防範內線交易管理程序"

    list_response = client.get("/cases")
    assert list_response.status_code == 200
    assert len(list_response.json()["cases"]) == 1

    get_response = client.get(f"/cases/{created['case_id']}")
    assert get_response.status_code == 200
    assert get_response.json()["case_id"] == created["case_id"]


def test_get_missing_case_returns_404():
    mock_case_store.clear_cases()

    response = client.get("/cases/CASE-MISSING")

    assert response.status_code == 404
    assert "Case not found" in response.json()["detail"]


def test_invalid_top_k_returns_validation_error():
    response = client.post("/qa", json={"query": "test", "top_k": 99})

    assert response.status_code == 422


def test_blank_query_is_rejected():
    response = client.post("/workflow-advice", json={"query": "   ", "top_k": 5})

    assert response.status_code == 422


def test_insufficient_evidence_returns_safe_structured_response(monkeypatch):
    insufficient = sample_workflow_response("這份文件有沒有說員工旅遊補助？")
    insufficient.update(
        {
            "risk_level": "Insufficient Evidence",
            "risk_category": "Unknown or Insufficient Evidence",
            "risk_reasoning": "The retrieved evidence is insufficient for confident triage.",
            "workflow_checklist": ["Do not rely on an automated workflow checklist for this question."],
            "evidence_quality": "The retrieved evidence is insufficient to answer this confidently.",
        }
    )
    monkeypatch.setattr(services, "build_workflow_response", lambda query, top_k=5: insufficient)

    response = client.post("/workflow-advice", json={"query": "這份文件有沒有說員工旅遊補助？", "top_k": 5})
    payload = response.json()

    assert response.status_code == 200
    assert payload["risk_level"] == "Insufficient Evidence"
    assert payload["risk_category"] == "Unknown or Insufficient Evidence"
    assert "Do not rely" in payload["workflow_checklist"][0]


def test_retrieval_service_error_returns_503(monkeypatch):
    def fail(query, top_k=5, use_llm=False):
        raise services.RetrievalServiceError("Local retrieval service is unavailable.")

    monkeypatch.setattr(services, "build_qa_response", fail)

    response = client.post("/qa", json={"query": "公司還沒公告重大資訊，可以買股票嗎？", "top_k": 5})

    assert response.status_code == 503
    assert "Local retrieval service is unavailable" in response.json()["detail"]
