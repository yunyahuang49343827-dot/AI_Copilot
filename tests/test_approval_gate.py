from src.agent.approval_gate import evaluate_approval_requirement


def test_high_risk_requires_approval_and_is_not_blocked():
    decision = evaluate_approval_requirement("High", "Insider Trading / Material Non-Public Information")

    assert decision.approval_required is True
    assert decision.approval_recommended is True
    assert decision.blocked is False
    assert decision.allowed_action == "create_mock_case_after_approval"
    assert "High-risk" in decision.reason


def test_medium_risk_recommends_approval_without_requiring_it():
    decision = evaluate_approval_requirement("Medium", "Asset Acquisition or Disposal")

    assert decision.approval_required is False
    assert decision.approval_recommended is True
    assert decision.blocked is False
    assert decision.allowed_action == "create_mock_case"


def test_low_risk_does_not_require_approval():
    decision = evaluate_approval_requirement("Low", "Ethical Conduct / Conflict of Interest")

    assert decision.approval_required is False
    assert decision.approval_recommended is False
    assert decision.blocked is False
    assert decision.allowed_action == "create_mock_case"


def test_insufficient_evidence_blocks_case_creation():
    decision = evaluate_approval_requirement("Insufficient Evidence", "Unknown or Insufficient Evidence")

    assert decision.approval_required is False
    assert decision.approval_recommended is False
    assert decision.blocked is True
    assert decision.allowed_action is None
    assert "blocked" in decision.reason.lower()


def test_unknown_risk_is_handled_conservatively():
    decision = evaluate_approval_requirement("")

    assert decision.approval_required is False
    assert decision.approval_recommended is False
    assert decision.blocked is True
    assert decision.allowed_action is None
    assert "manual review" in decision.reason.lower()
