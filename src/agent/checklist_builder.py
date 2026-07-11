"""Workflow checklist generation for Day 7 risk categories."""

from __future__ import annotations

from typing import Any, Callable

from src.agent.risk_triage import (
    CATEGORY_ASSET,
    CATEGORY_BOARD,
    CATEGORY_DERIVATIVE,
    CATEGORY_ENDORSEMENT,
    CATEGORY_ETHICS,
    CATEGORY_FUNDS,
    CATEGORY_INSIDER,
    CATEGORY_RELATED_PARTY,
    CATEGORY_SUSTAINABILITY,
    CATEGORY_UNKNOWN,
    CATEGORY_WHISTLEBLOWING,
    RISK_INSUFFICIENT,
    RiskTriageResult,
)


def first_label(evidence_rows: list[tuple[str, dict[str, Any]]], predicate: Callable[[dict[str, Any]], bool]) -> str:
    for label, result in evidence_rows:
        if predicate(result):
            return label
    return ""


def policy_or_term_label(evidence_rows: list[tuple[str, dict[str, Any]]], values: list[str]) -> str:
    return first_label(
        evidence_rows,
        lambda result: any(
            value in " ".join(str(result.get(field, "")) for field in ["policy_name", "article_title", "text", "text_preview"])
            for value in values
        ),
    )


def with_label(text: str, label: str) -> str:
    return f"{text} {label}" if label else text


def build_checklist(question: str, triage: RiskTriageResult, evidence_rows: list[tuple[str, dict[str, Any]]]) -> list[str]:
    if triage.risk_level == RISK_INSUFFICIENT or triage.risk_category == CATEGORY_UNKNOWN:
        return [
            "Do not rely on an automated workflow checklist for this question.",
            "Manually review the relevant policy documents or ask a responsible human reviewer to confirm whether a policy applies.",
            "Record the question, retrieved evidence, and reason the evidence was considered insufficient.",
        ]

    category = triage.risk_category
    insider = policy_or_term_label(evidence_rows, ["防範內線交易管理程序", "內部重大資訊處理作業程序", "未公開", "重大資訊", "重大消息"])
    whistle = policy_or_term_label(evidence_rows, ["舉報制度", "檢舉", "舉報", "具體事證", "保密"])
    related = policy_or_term_label(evidence_rows, ["關係人交易管理辦法", "關係人", "特定公司", "集團企業"])
    board = policy_or_term_label(evidence_rows, ["董事會", "決議", "提交董事會", "獨立董事"])
    derivative = policy_or_term_label(evidence_rows, ["從事衍生性商品交易處理規範", "交易策略", "避險", "非避險", "Forward", "Option", "Swap", "Future"])
    asset = policy_or_term_label(evidence_rows, ["取得或處分資產處理程序", "資產", "合併", "分割", "收購"])
    funds = policy_or_term_label(evidence_rows, ["資金貸與"])
    endorsement = policy_or_term_label(evidence_rows, ["背書保證"])

    if category == CATEGORY_INSIDER:
        return [
            with_label("Do not trade, tip, or share the information until compliance review is completed.", insider),
            with_label("Confirm whether the facts are internal material information or major news under the retrieved policies.", insider),
            with_label("Check whether the information has been publicly disclosed and whether any timing restriction applies.", insider),
            "Escalate to the responsible compliance/legal unit for human review before any trading decision.",
            "Keep records of the facts, timing, people involved, and evidence reviewed.",
        ]

    if category == CATEGORY_WHISTLEBLOWING:
        return [
            with_label("Prepare reporter identity and contact information if the retrieved policy requires it.", whistle),
            with_label("Identify the reported person or provide information sufficient to identify them.", whistle),
            with_label("Prepare concrete facts and evidence that can support investigation.", whistle),
            with_label("Submit through the appropriate reporting channel described by the retrieved policy evidence.", whistle),
            with_label("Treat reporter identity and report content as confidential.", whistle),
            "Escalate to a responsible human reviewer if the case involves management, bribery, fraud, retaliation, or legal exposure.",
        ]

    if category == CATEGORY_DERIVATIVE:
        return [
            with_label("Confirm the transaction type, such as Forward, Option, Swap, or Future.", derivative),
            with_label("Confirm whether the transaction purpose is hedging rather than speculation or non-hedging trading.", derivative),
            with_label("Check whether the transaction fits the retrieved trading strategy and authority evidence.", derivative),
            with_label("Confirm whether board approval or special approval is needed for other products.", board or derivative),
            "Record risk assessment, approval, and post-transaction monitoring requirements.",
            "Escalate to finance/compliance review before execution.",
        ]

    if category == CATEGORY_RELATED_PARTY:
        return [
            with_label("Identify the counterparty and determine whether it is a related party, specific company, or group enterprise.", related),
            with_label("Identify transaction type, purpose, necessity, expected benefit, and amount.", related),
            with_label("Check whether board approval or board resolution is required under the retrieved policy evidence.", board or related),
            with_label("Record independent director opinions when applicable.", board or related),
            "Escalate for governance, audit committee, or legal review if the amount or conflict risk is material.",
        ]

    if category == CATEGORY_BOARD:
        return [
            with_label("Identify the decision or transaction requiring board review.", board),
            with_label("Confirm whether the retrieved evidence requires board approval, resolution, or submission to the board.", board),
            with_label("Prepare supporting materials for the board decision.", board),
            "Record approvals, dissenting opinions, and follow-up owners.",
            "Escalate to governance/legal review for medium or high-risk matters.",
        ]

    if category == CATEGORY_ASSET:
        return [
            with_label("Identify the asset transaction type and transaction counterparties.", asset),
            with_label("Confirm whether board or shareholder approval is required by the retrieved evidence.", board or asset),
            with_label("Prepare transaction purpose, necessity, valuation, and expected benefit for review.", asset),
            "Escalate to finance/legal review for material transactions.",
        ]

    if category == CATEGORY_FUNDS:
        return [
            with_label("Confirm whether the case involves funds lending to another party.", funds),
            with_label("Identify borrower, amount, purpose, term, and repayment source.", funds),
            with_label("Check approval authority and disclosure requirements in the retrieved evidence.", funds),
            "Escalate to finance/legal review before execution.",
        ]

    if category == CATEGORY_ENDORSEMENT:
        return [
            with_label("Confirm whether the case involves endorsement or guarantee obligations.", endorsement),
            with_label("Identify beneficiary, amount, purpose, and risk exposure.", endorsement),
            with_label("Check approval authority and disclosure requirements in the retrieved evidence.", endorsement),
            "Escalate to finance/legal review before execution.",
        ]

    if category in {CATEGORY_ETHICS, CATEGORY_SUSTAINABILITY}:
        label = policy_or_term_label(evidence_rows, ["道德", "誠信", "永續", "風險管理", "公司治理"])
        return [
            with_label("Identify the policy topic and affected stakeholders.", label),
            with_label("Check the retrieved policy evidence for required conduct, disclosure, or escalation expectations.", label),
            "Document facts, owners, timing, and open questions.",
            "Escalate medium or high-risk matters for human review.",
        ]

    return [
        "Manually review the relevant policy evidence before taking action.",
        "Escalate to a responsible human reviewer if the matter could affect compliance, governance, or financial controls.",
    ]
