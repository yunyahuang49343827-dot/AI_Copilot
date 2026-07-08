from src.agent.checklist_builder import build_checklist
from src.agent.risk_triage import (
    CATEGORY_DERIVATIVE,
    CATEGORY_INSIDER,
    CATEGORY_UNKNOWN,
    RISK_HIGH,
    RISK_INSUFFICIENT,
    RiskTriageResult,
)


def triage(category=CATEGORY_INSIDER, level=RISK_HIGH):
    return RiskTriageResult(level, category, "reason [C1]", ["[C1]"])


def row(label="[C1]", **overrides):
    result = {
        "policy_name": "防範內線交易管理程序",
        "article_title": "重大消息",
        "text": "重大資訊未公開前，不得洩露或買賣有價證券。",
        "text_preview": "重大資訊未公開前，不得洩露或買賣有價證券。",
    }
    result.update(overrides)
    return label, result


def test_supported_category_checklist_has_4_to_7_steps():
    checklist = build_checklist("公司還沒公告重大資訊，可以買股票嗎？", triage(), [row()])

    assert 4 <= len(checklist) <= 7


def test_checklist_citations_only_use_supplied_labels():
    evidence = [row("[C7]"), row("[C9]", policy_name="內部重大資訊處理作業程序")]

    checklist = build_checklist("公司還沒公告重大資訊，可以買股票嗎？", triage(), evidence)
    joined = " ".join(checklist)

    assert "[C7]" in joined or "[C9]" in joined
    assert "[C1]" not in joined
    assert "[C2]" not in joined


def test_weak_evidence_does_not_generate_confident_steps():
    weak = RiskTriageResult(RISK_INSUFFICIENT, CATEGORY_UNKNOWN, "weak", [])

    checklist = build_checklist("員工旅遊補助", weak, [])

    assert len(checklist) == 3
    assert "Do not rely on an automated workflow checklist" in checklist[0]
    assert not any("board approval" in step.lower() for step in checklist)


def test_derivative_checklist_includes_hedging_and_review_steps():
    evidence = [row("[C3]", policy_name="從事衍生性商品交易處理規範", article_title="交易策略", text="應以避險策略為原則，不得從事非避險性交易。")]

    checklist = build_checklist("衍生性商品交易可以投機嗎？", triage(CATEGORY_DERIVATIVE), evidence)
    joined = " ".join(checklist)

    assert "hedging" in joined
    assert "finance/compliance" in joined
