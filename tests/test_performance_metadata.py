from backend import agent_run_store, audit_log_store, mock_case_store, services
from backend.performance import elapsed_ms, start_timer
from src.agent.human_loop_agent import start_agent_run


def sample_retrieved_result():
    return {
        "chunk_id": "TEST-001-article-005",
        "document_id": "WITS-001",
        "policy_name": "防範內線交易管理程序",
        "article": "第五條",
        "article_title": "重大消息",
        "page_start": 1,
        "page_end": 2,
        "source_file": "policy.pdf",
        "text": "公司重大資訊尚未公告前，知悉內部重大資訊之人不得買賣公司股票或有價證券。",
        "text_preview": "公司重大資訊尚未公告前，知悉內部重大資訊之人不得買賣公司股票或有價證券。",
        "final_score": 0.95,
        "keyword_boost": 0.2,
        "policy_boost": 0.2,
        "boilerplate_penalty": 0.0,
    }


def assert_non_negative_timing(metadata, expected_keys):
    assert isinstance(metadata, dict)
    assert set(metadata) == set(expected_keys)
    for key in expected_keys:
        assert isinstance(metadata[key], float)
        assert metadata[key] >= 0


def test_performance_helpers_return_elapsed_milliseconds():
    start = start_timer()

    duration = elapsed_ms(start)

    assert isinstance(duration, float)
    assert duration >= 0


def test_qa_response_includes_service_timing_metadata_without_real_retrieval():
    response = services.build_qa_response(
        "公司還沒公告重大資訊，可以買股票嗎？",
        top_k=5,
        retrieved_results=[sample_retrieved_result()],
    )

    assert_non_negative_timing(
        response["timing_metadata"],
        ["retrieval_time_ms", "answer_time_ms", "total_time_ms"],
    )
    assert response["answer"]
    assert response["citations"][0]["chunk_id"] == "TEST-001-article-005"


def test_workflow_response_includes_service_timing_metadata_without_real_retrieval():
    response = services.build_workflow_response(
        "公司還沒公告重大資訊，可以買股票嗎？",
        top_k=5,
        retrieved_results=[sample_retrieved_result()],
    )

    assert_non_negative_timing(
        response["timing_metadata"],
        ["retrieval_time_ms", "workflow_time_ms", "total_time_ms"],
    )
    assert response["risk_level"] == "High"
    assert response["workflow_checklist"]


def test_agent_run_service_includes_timing_metadata_and_keeps_audit_metadata_sanitized(monkeypatch):
    agent_run_store.clear_runs()
    audit_log_store.clear_audit_events()
    mock_case_store.clear_cases()

    def fake_start_agent_run(query, requester="Demo User", department="Compliance", top_k=5):
        return start_agent_run(
            query,
            requester=requester,
            department=department,
            top_k=top_k,
            workflow_func=lambda workflow_query, workflow_top_k: services.build_workflow_response(
                workflow_query,
                top_k=workflow_top_k,
                retrieved_results=[sample_retrieved_result()],
            ),
        )

    monkeypatch.setattr(services, "start_agent_run", fake_start_agent_run)

    response = services.start_agent_run_service(
        "公司還沒公告重大資訊，可以買股票嗎？",
        requester="demo_user",
        department="Finance",
        top_k=5,
    )

    assert_non_negative_timing(
        response["timing_metadata"],
        ["agent_run_time_ms", "store_time_ms", "audit_time_ms", "total_time_ms"],
    )
    assert response["approval_required"] is True

    events = audit_log_store.list_audit_events(run_id=response["run_id"])
    assert events
    for event in events:
        assert "timing_metadata" not in event["metadata"]
        assert "total_time_ms" not in event["metadata"]
