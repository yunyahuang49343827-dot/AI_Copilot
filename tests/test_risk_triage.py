from src.agent.risk_triage import (
    CATEGORY_ASSET,
    CATEGORY_DERIVATIVE,
    CATEGORY_INSIDER,
    CATEGORY_RELATED_PARTY,
    CATEGORY_UNKNOWN,
    CATEGORY_WHISTLEBLOWING,
    RISK_HIGH,
    RISK_INSUFFICIENT,
    RISK_MEDIUM,
    classify_risk,
)
from src.guardrails.evidence_checks import EvidenceAssessment


def assessment(is_sufficient=True):
    return EvidenceAssessment(is_sufficient, "note", [])


def row(label="[C1]", **overrides):
    result = {
        "chunk_id": "chunk-1",
        "policy_name": "防範內線交易管理程序",
        "article": "第五條",
        "article_title": "重大消息",
        "text": "重大資訊未公開前，不得洩露或買賣有價證券。",
        "text_preview": "重大資訊未公開前，不得洩露或買賣有價證券。",
    }
    result.update(overrides)
    return label, result


def test_insider_material_information_query_returns_high_risk():
    triage = classify_risk("公司還沒公告重大資訊，可以買股票嗎？", [row()], assessment())

    assert triage.risk_level == RISK_HIGH
    assert triage.risk_category == CATEGORY_INSIDER
    assert "[C1]" in triage.reasoning


def test_derivative_speculation_query_returns_high_risk():
    evidence = [row(policy_name="從事衍生性商品交易處理規範", article_title="交易策略", text="不得從事非避險性交易，應以避險策略為原則。")]

    triage = classify_risk("衍生性商品交易可以投機嗎？", evidence, assessment())

    assert triage.risk_level == RISK_HIGH
    assert triage.risk_category == CATEGORY_DERIVATIVE


def test_whistleblowing_query_returns_whistleblowing_category():
    evidence = [row(policy_name="舉報制度", article_title="保障及保密", text="檢舉人應提供具體事證，公司應保密處理舉報資料。")]

    triage = classify_risk("我想檢舉內部舞弊，應該提供哪些資料？", evidence, assessment())

    assert triage.risk_category == CATEGORY_WHISTLEBLOWING
    assert triage.risk_level == RISK_HIGH


def test_related_party_query_returns_related_party_category():
    evidence = [row(policy_name="與特定公司集團企業及關係人交易管理辦法", article="第十五條", text="關係人交易達一定金額者，應提交董事會通過。")]

    triage = classify_risk("關係人交易需要董事會核准嗎？", evidence, assessment())

    assert triage.risk_category == CATEGORY_RELATED_PARTY


def test_weak_evidence_returns_insufficient_evidence():
    triage = classify_risk("這份文件有沒有說員工旅遊補助？", [row(policy_name="無關政策", text="員工溝通管道。")], assessment(False))

    assert triage.risk_level == RISK_INSUFFICIENT
    assert triage.risk_category == CATEGORY_UNKNOWN


def test_financial_asset_procedure_query_returns_asset_category_and_medium_risk():
    evidence = [
        row(
            policy_name="取得或處分資產處理程序",
            document_id="WITS-004",
            article_title="本處理程序所稱「資產」之適用範圍如下",
            text="本處理程序所稱資產包括有價證券、金融資產、不動產及設備。",
            text_preview="本處理程序所稱資產包括有價證券、金融資產、不動產及設備。",
        )
    ]

    triage = classify_risk("取得有價證券或金融資產時，應該參考哪個作業程序？", evidence, assessment())

    assert triage.risk_category == CATEGORY_ASSET
    assert triage.risk_level == RISK_MEDIUM


def test_securities_in_asset_context_does_not_trigger_insider_category():
    evidence = [
        row(
            policy_name="取得或處分資產處理程序",
            document_id="WITS-004",
            article_title="取得或處分資產評估及作業程序",
            text="取得有價證券應依資產處理程序評估並辦理。",
            text_preview="取得有價證券應依資產處理程序評估並辦理。",
        )
    ]

    triage = classify_risk("取得有價證券時，應該參考哪個作業程序？", evidence, assessment())

    assert triage.risk_category == CATEGORY_ASSET
    assert triage.risk_category != CATEGORY_INSIDER


def test_insider_material_information_regression_still_returns_high_risk():
    triage = classify_risk("公司還沒公告重大資訊，可以買股票嗎？", [row()], assessment())

    assert triage.risk_level == RISK_HIGH
    assert triage.risk_category == CATEGORY_INSIDER
