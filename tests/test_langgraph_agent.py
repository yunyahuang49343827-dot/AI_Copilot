from src.agent.agent_state import AgentState
from src.agent.langgraph_agent import graph_result_to_agent_state, start_langgraph_agent_run


def workflow_response(risk_level="High", risk_category="Insider Trading / Material Non-Public Information"):
    return {
        "query": "query",
        "grounded_answer": "FULL EVIDENCE TEXT SHOULD NOT APPEAR IN TRACE.",
        "risk_level": risk_level,
        "risk_category": risk_category,
        "risk_reasoning": "The query matches a controlled policy category. [C1]",
        "workflow_checklist": ["Review the matter. [C1]", "Escalate when required."],
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


def workflow_func(response):
    calls = []

    def run(query, top_k):
        calls.append({"query": query, "top_k": top_k})
        return dict(response, query=query)

    run.calls = calls
    return run


def test_high_risk_graph_run_reaches_pending_approval_and_creates_no_case():
    mocked_workflow = workflow_func(workflow_response("High"))

    result = start_langgraph_agent_run(
        "公司還沒公告重大資訊，可以買股票嗎？",
        workflow_func=mocked_workflow,
    )

    assert result["route"] == "pending_approval"
    assert result["pending_action"]["action_type"] == "create_mock_case"
    assert result["pending_action"]["requires_approval"] is True
    assert result["decision"]["approval_required"] is True
    assert result["case_created"] is False
    assert result["case_id"] is None
    assert len(mocked_workflow.calls) == 1


def test_insufficient_evidence_graph_run_is_blocked_and_creates_no_case():
    result = start_langgraph_agent_run(
        "員工旅遊補助可以報銷嗎？",
        workflow_func=workflow_func(workflow_response("Insufficient Evidence", "Unknown or Insufficient Evidence")),
    )

    assert result["route"] == "blocked"
    assert result["pending_action"]["action_type"] == "manual_review"
    assert result["pending_action"]["blocked"] is True
    assert result["decision"]["blocked"] is True
    assert result["case_created"] is False
    assert result["case_id"] is None


def test_medium_risk_graph_run_is_ready_for_confirmation_without_auto_case_creation():
    result = start_langgraph_agent_run(
        "取得有價證券或金融資產時，應該參考哪個作業程序？",
        workflow_func=workflow_func(workflow_response("Medium", "Asset Acquisition or Disposal")),
    )

    assert result["route"] == "ready_for_confirmation"
    assert result["decision"]["approval_recommended"] is True
    assert result["decision"]["approval_required"] is False
    assert result["pending_action"]["requires_approval"] is False
    assert result["case_created"] is False
    assert result["case_id"] is None


def test_low_risk_graph_run_is_ready_for_confirmation_without_auto_case_creation():
    result = start_langgraph_agent_run(
        "一般利益衝突申報流程是什麼？",
        workflow_func=workflow_func(workflow_response("Low", "Ethical Conduct / Conflict of Interest")),
    )

    assert result["route"] == "ready_for_confirmation"
    assert result["decision"]["approval_required"] is False
    assert result["decision"]["blocked"] is False
    assert result["case_created"] is False
    assert result["case_id"] is None


def test_graph_trace_exists_and_does_not_include_full_evidence_text():
    result = start_langgraph_agent_run("query", workflow_func=workflow_func(workflow_response("High")))

    steps = [entry["step"] for entry in result["trace"]]
    messages = " ".join(entry["message"] for entry in result["trace"])

    assert "intake" in steps
    assert "workflow" in steps
    assert "approval_gate" in steps
    assert "final" in steps
    assert "FULL EVIDENCE TEXT SHOULD NOT APPEAR IN TRACE" not in messages
    assert "Evidence quality appears sufficient" not in messages


def test_graph_result_to_agent_state_returns_json_friendly_agent_state():
    result = start_langgraph_agent_run(
        "query",
        requester="Reviewer",
        department="Legal",
        top_k=3,
        workflow_func=workflow_func(workflow_response("Medium", "Asset Acquisition or Disposal")),
    )

    state = graph_result_to_agent_state(result)
    data = state.to_dict()

    assert isinstance(state, AgentState)
    assert state.query == "query"
    assert state.requester == "Reviewer"
    assert state.department == "Legal"
    assert state.top_k == 3
    assert state.case_created is False
    assert state.case_id is None
    assert isinstance(data["decision"], dict)
    assert isinstance(data["pending_action"], dict)
    assert isinstance(data["trace"], list)
    assert data["case_created"] is False
