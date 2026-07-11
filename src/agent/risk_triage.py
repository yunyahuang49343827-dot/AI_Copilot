"""Rule-based risk triage for Day 7 workflow advice."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.guardrails.evidence_checks import EvidenceAssessment


RISK_HIGH = "High"
RISK_MEDIUM = "Medium"
RISK_LOW = "Low"
RISK_INSUFFICIENT = "Insufficient Evidence"

CATEGORY_INSIDER = "Insider Trading / Material Non-Public Information"
CATEGORY_DISCLOSURE = "Internal Material Information / Disclosure"
CATEGORY_WHISTLEBLOWING = "Whistleblowing / Fraud Reporting"
CATEGORY_RELATED_PARTY = "Related-Party Transaction"
CATEGORY_BOARD = "Board Approval / Governance Procedure"
CATEGORY_DERIVATIVE = "Derivative Trading / Hedging Control"
CATEGORY_ASSET = "Asset Acquisition or Disposal"
CATEGORY_FUNDS = "Funds Lending"
CATEGORY_ENDORSEMENT = "Endorsement and Guarantee"
CATEGORY_ETHICS = "Ethical Conduct / Conflict of Interest"
CATEGORY_SUSTAINABILITY = "Sustainability / Risk Management"
CATEGORY_UNKNOWN = "Unknown or Insufficient Evidence"

CATEGORY_PRECEDENCE = [
    CATEGORY_INSIDER,
    CATEGORY_WHISTLEBLOWING,
    CATEGORY_DERIVATIVE,
    CATEGORY_RELATED_PARTY,
    CATEGORY_ASSET,
    CATEGORY_FUNDS,
    CATEGORY_ENDORSEMENT,
    CATEGORY_BOARD,
    CATEGORY_ETHICS,
    CATEGORY_SUSTAINABILITY,
    CATEGORY_UNKNOWN,
]

CATEGORY_TERMS = {
    CATEGORY_INSIDER: ["內線交易", "重大資訊", "重大消息", "未公開", "買股票", "買賣股票", "十八小時", "有價證券"],
    CATEGORY_DISCLOSURE: ["重大資訊", "重大消息", "公開資訊", "揭露", "公告"],
    CATEGORY_WHISTLEBLOWING: ["檢舉", "舉報", "通報", "申訴", "舞弊", "不法", "違規", "違反誠信", "行賄"],
    CATEGORY_RELATED_PARTY: ["關係人交易", "關係企業", "特定公司", "集團企業", "關係人"],
    CATEGORY_BOARD: ["董事會", "核准", "決議", "提交董事會", "獨立董事"],
    CATEGORY_DERIVATIVE: ["衍生性商品", "投機", "非避險", "非避險性交易", "避險", "Forward", "Option", "Swap", "Future"],
    CATEGORY_ASSET: ["取得或處分資產", "資產交易", "合併", "分割", "收購", "股份受讓", "處分資產"],
    CATEGORY_FUNDS: ["資金貸與"],
    CATEGORY_ENDORSEMENT: ["背書保證"],
    CATEGORY_ETHICS: ["道德", "誠信", "利益衝突", "收受", "餽贈", "禮品", "招待"],
    CATEGORY_SUSTAINABILITY: ["永續", "風險管理", "公司治理", "環境", "社會"],
}

CATEGORY_POLICY_HINTS = {
    CATEGORY_INSIDER: ["防範內線交易管理程序", "內部重大資訊處理作業程序"],
    CATEGORY_DISCLOSURE: ["內部重大資訊處理作業程序"],
    CATEGORY_WHISTLEBLOWING: ["舉報制度", "道德行為準則", "誠信經營守則"],
    CATEGORY_RELATED_PARTY: ["與特定公司集團企業及關係人交易管理辦法"],
    CATEGORY_DERIVATIVE: ["從事衍生性商品交易處理規範"],
    CATEGORY_ASSET: ["取得或處分資產處理程序"],
    CATEGORY_FUNDS: ["資金貸與他人作業程序"],
    CATEGORY_ENDORSEMENT: ["背書保證作業程序"],
    CATEGORY_ETHICS: ["道德行為準則", "誠信經營守則"],
    CATEGORY_SUSTAINABILITY: ["風險管理政策與程序", "永續發展實務守則", "公司治理實務守則"],
}

INSIDER_CONTEXT_TERMS = [
    "內線交易",
    "重大資訊",
    "重大消息",
    "未公開",
    "尚未公告",
    "公告前",
    "買股票",
    "賣股票",
    "買賣股票",
    "交易公司股票",
    "十八小時",
    "獲悉",
    "內部重大資訊",
]
ASSET_CONTEXT_TERMS = [
    "資產",
    "資產交易",
    "取得",
    "處分",
    "出售",
    "不動產",
    "設備",
    "金融資產",
    "有價證券",
    "價格合理性",
    "估價",
    "作業程序",
    "取得或處分資產",
]
ASSET_HIGH_RISK_TERMS = ["董事會", "重大", "大額", "公告", "核准", "違規", "未公開", "重大資訊", "內線交易"]

CATEGORY_TERMS[CATEGORY_INSIDER] = INSIDER_CONTEXT_TERMS
CATEGORY_TERMS[CATEGORY_ASSET] = ASSET_CONTEXT_TERMS


@dataclass(frozen=True)
class RiskTriageResult:
    risk_level: str
    risk_category: str
    reasoning: str
    matched_labels: list[str]


def combined_evidence_text(evidence_rows: list[tuple[str, dict[str, Any]]]) -> str:
    return " ".join(
        " ".join(str(result.get(field, "")) for field in ["policy_name", "article_title", "article", "text", "text_preview"])
        for _, result in evidence_rows
    )


def labels_for_category(category: str, evidence_rows: list[tuple[str, dict[str, Any]]]) -> list[str]:
    terms = CATEGORY_TERMS.get(category, [])
    policies = CATEGORY_POLICY_HINTS.get(category, [])
    labels: list[str] = []
    for label, result in evidence_rows:
        haystack = " ".join(str(result.get(field, "")) for field in ["policy_name", "article_title", "text", "text_preview"])
        if any(policy in haystack for policy in policies) or any(term in haystack for term in terms):
            labels.append(label)
    return labels[:3]


def query_has(question: str, category: str) -> bool:
    return any(term in question for term in CATEGORY_TERMS.get(category, []))


def evidence_has(evidence_rows: list[tuple[str, dict[str, Any]]], category: str) -> bool:
    text = combined_evidence_text(evidence_rows)
    return any(policy in text for policy in CATEGORY_POLICY_HINTS.get(category, [])) or any(term in text for term in CATEGORY_TERMS.get(category, []))


def matched_categories(question: str, evidence_rows: list[tuple[str, dict[str, Any]]]) -> list[str]:
    matches: list[str] = []
    for category in CATEGORY_PRECEDENCE:
        if category == CATEGORY_UNKNOWN:
            continue
        if query_has(question, category) and evidence_has(evidence_rows, category):
            matches.append(category)
    return matches


def query_has_asset_context(question: str) -> bool:
    return any(term in question for term in ASSET_CONTEXT_TERMS)


def evidence_has_asset_policy(evidence_rows: list[tuple[str, dict[str, Any]]]) -> bool:
    return any(
        result.get("document_id") == "WITS-004" or result.get("policy_name") == "取得或處分資產處理程序"
        for _, result in evidence_rows
    )


def has_insider_context(question: str, evidence_rows: list[tuple[str, dict[str, Any]]]) -> bool:
    text = f"{question} {combined_evidence_text(evidence_rows)}"
    return any(term in text for term in INSIDER_CONTEXT_TERMS)


def choose_category(question: str, evidence_rows: list[tuple[str, dict[str, Any]]]) -> str:
    if query_has_asset_context(question) and evidence_has_asset_policy(evidence_rows):
        return CATEGORY_ASSET

    matches = matched_categories(question, evidence_rows)
    if CATEGORY_INSIDER in matches and not has_insider_context(question, evidence_rows):
        matches = [category for category in matches if category != CATEGORY_INSIDER]
    if matches:
        return sorted(matches, key=CATEGORY_PRECEDENCE.index)[0]
    if "董事會" in question or "核准" in question:
        if evidence_has(evidence_rows, CATEGORY_BOARD):
            return CATEGORY_BOARD
    return CATEGORY_UNKNOWN


def classify_level(question: str, category: str, evidence_rows: list[tuple[str, dict[str, Any]]]) -> str:
    text = f"{question} {combined_evidence_text(evidence_rows)}"
    if category == CATEGORY_INSIDER:
        return RISK_HIGH
    if category == CATEGORY_DERIVATIVE:
        return RISK_HIGH if any(term in question for term in ["投機", "非避險", "非避險性交易"]) else RISK_MEDIUM
    if category == CATEGORY_WHISTLEBLOWING:
        high_terms = ["舞弊", "行賄", "賄賂", "不法", "報復", "保密", "法律", "主管", "管理階層"]
        return RISK_HIGH if any(term in text for term in high_terms) else RISK_MEDIUM
    if category == CATEGORY_RELATED_PARTY:
        high_terms = ["重大", "大額", "資產", "董事會", "核准", "利益衝突", "揭露", "公告", "獨立董事"]
        return RISK_HIGH if any(term in text for term in high_terms) else RISK_MEDIUM
    if category == CATEGORY_ASSET:
        return RISK_HIGH if any(term in text for term in ASSET_HIGH_RISK_TERMS) else RISK_MEDIUM
    if category in {CATEGORY_FUNDS, CATEGORY_ENDORSEMENT}:
        return RISK_HIGH if any(term in text for term in ["董事會", "重大", "大額", "公告", "核准"]) else RISK_MEDIUM
    if category in {CATEGORY_BOARD, CATEGORY_ETHICS, CATEGORY_SUSTAINABILITY, CATEGORY_DISCLOSURE}:
        return RISK_MEDIUM
    return RISK_LOW


def classify_risk(question: str, evidence_rows: list[tuple[str, dict[str, Any]]], evidence_assessment: EvidenceAssessment) -> RiskTriageResult:
    if not evidence_assessment.is_sufficient or not evidence_rows:
        return RiskTriageResult(
            risk_level=RISK_INSUFFICIENT,
            risk_category=CATEGORY_UNKNOWN,
            reasoning="The retrieved evidence is insufficient for confident triage. Manual policy review or human review is required.",
            matched_labels=[],
        )

    category = choose_category(question, evidence_rows)
    if category == CATEGORY_UNKNOWN:
        return RiskTriageResult(
            risk_level=RISK_INSUFFICIENT,
            risk_category=CATEGORY_UNKNOWN,
            reasoning="No supported risk category could be assigned from the retrieved evidence. Manual policy review or human review is required.",
            matched_labels=[],
        )

    level = classify_level(question, category, evidence_rows)
    labels = labels_for_category(category, evidence_rows)
    label_text = "".join(labels)
    reasoning = f"The query and retrieved policy evidence match {category}; triage priority is {level}."
    if label_text:
        reasoning += f" {label_text}"
    return RiskTriageResult(level, category, reasoning, labels)
