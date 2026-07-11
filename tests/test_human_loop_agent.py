from src.agent.human_loop_agent import approve_agent_action, reject_agent_action, start_agent_run


def workflow_response(risk_level="High", risk_category="Insider Trading / Material Non-Public Information"):
    return {
        "query": "query",
        "grounded_answer": "Full answer should stay out of trace.",
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


def workflow_func(response):
    return lambda query, top_k: dict(response, query=query)


class CaseCreator:
    def __init__(self):
        self.calls = []

    def __call__(self, case_record):
        self.calls.append(case_record)
        return {
            "case_id": f"CASE-{len(self.calls)}",
            "status": "Created",
            "created_at": "2026-07-07T00:00:00+00:00",
            **case_record,
        }


def test_high_risk_start_agent_run_returns_pending_action_and_no_case():
    state = start_agent_run(
        "公司還沒公告重大資訊，可以買股票嗎？",
        workflow_func=workflow_func(workflow_response("High")),
    )

    assert state.decision.approval_required is True
    assert state.pending_action is not None
    assert state.pending_action.action_type == "create_mock_case"
    assert state.pending_action.requires_approval is True
    assert state.case_created is False
    assert state.case_id is None


def test_approving_high_risk_pending_action_creates_exactly_one_mock_case():
    creator = CaseCreator()
    state = start_agent_run("query", workflow_func=workflow_func(workflow_response("High")))

    approved = approve_agent_action(state, approver="Reviewer", case_creator=creator)

    assert approved.human_decision == "approved"
    assert approved.case_created is True
    assert approved.case_id == "CASE-1"
    assert len(creator.calls) == 1
    assert creator.calls[0]["agent_decision"]["approval_required"] is True


def test_rejecting_high_risk_pending_action_creates_no_case_and_records_rejection():
    state = start_agent_run("query", workflow_func=workflow_func(workflow_response("High")))

    rejected = reject_agent_action(state, reviewer="Reviewer", reason="Need more details.")

    assert rejected.human_decision == "rejected"
    assert rejected.case_created is False
    assert rejected.case_id is None
    assert any(entry.status == "rejected" and "Need more details" in entry.message for entry in rejected.trace)


def test_insufficient_evidence_is_blocked_and_approve_cannot_create_case():
    creator = CaseCreator()
    state = start_agent_run(
        "員工旅遊補助",
        workflow_func=workflow_func(workflow_response("Insufficient Evidence", "Unknown or Insufficient Evidence")),
    )

    approved = approve_agent_action(state, case_creator=creator)

    assert state.decision.blocked is True
    assert approved.human_decision == "blocked"
    assert approved.case_created is False
    assert approved.case_id is None
    assert len(creator.calls) == 0


def test_medium_risk_recommends_approval_but_does_not_require_it():
    state = start_agent_run(
        "取得有價證券或金融資產時，應該參考哪個作業程序？",
        workflow_func=workflow_func(workflow_response("Medium", "Asset Acquisition or Disposal")),
    )

    assert state.decision.approval_required is False
    assert state.decision.approval_recommended is True
    assert state.pending_action is not None
    assert state.pending_action.requires_approval is False


def test_medium_risk_approve_action_can_create_mock_case():
    creator = CaseCreator()
    state = start_agent_run("query", workflow_func=workflow_func(workflow_response("Medium", "Asset Acquisition or Disposal")))

    approved = approve_agent_action(state, case_creator=creator)

    assert approved.case_created is True
    assert approved.case_id == "CASE-1"
    assert len(creator.calls) == 1
    assert creator.calls[0]["risk_level"] == "Medium"


def test_low_risk_can_create_mock_case_after_positive_confirmation():
    creator = CaseCreator()
    state = start_agent_run("query", workflow_func=workflow_func(workflow_response("Low", "Ethical Conduct / Conflict of Interest")))

    approved = approve_agent_action(state, case_creator=creator)

    assert state.decision.approval_required is False
    assert state.decision.approval_recommended is False
    assert approved.case_created is True
    assert approved.case_id == "CASE-1"


def test_approve_agent_action_is_idempotent():
    creator = CaseCreator()
    state = start_agent_run("query", workflow_func=workflow_func(workflow_response("High")))

    approve_agent_action(state, case_creator=creator)
    approve_agent_action(state, case_creator=creator)

    assert state.case_created is True
    assert state.case_id == "CASE-1"
    assert len(creator.calls) == 1
    assert any(entry.status == "skipped" for entry in state.trace)


def test_agent_state_to_dict_returns_json_friendly_structure():
    state = start_agent_run("query", workflow_func=workflow_func(workflow_response("Medium", "Asset Acquisition or Disposal")))

    data = state.to_dict()

    assert data["query"] == "query"
    assert isinstance(data["workflow_response"], dict)
    assert isinstance(data["decision"], dict)
    assert isinstance(data["pending_action"], dict)
    assert isinstance(data["trace"], list)
    assert isinstance(data["trace"][0], dict)


def test_trace_entries_exist_and_do_not_include_full_evidence_text():
    state = start_agent_run("query", workflow_func=workflow_func(workflow_response("High")))

    messages = " ".join(entry.message for entry in state.trace)

    assert len(state.trace) >= 4
    assert "Full answer should stay out of trace" not in messages
    assert "Evidence quality appears sufficient" not in messages
