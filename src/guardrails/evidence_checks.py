"""Evidence quality checks for grounded policy Q&A."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.retrieval.bm25_store import tokenize


MIN_TOP_SCORE = 0.40
BOILERPLATE_RATIO_LIMIT = 0.60
REQUIRED_CITATION_FIELDS = ["chunk_id", "policy_name", "article", "page_start", "page_end", "source_file"]
ASSET_POLICY_ID = "WITS-004"
ASSET_POLICY_NAME = "取得或處分資產處理程序"
ASSET_TERMS = [
    "資產",
    "資產交易",
    "取得或處分資產",
    "取得",
    "處分",
    "出售",
    "不動產",
    "設備",
    "金融資產",
    "有價證券",
    "評估",
    "價格合理性",
    "估價",
    "作業程序",
]

SYNONYMS = {
    "檢舉": ["檢舉", "舉報", "通報", "申訴"],
    "舉報": ["檢舉", "舉報", "通報", "申訴"],
    "舞弊": ["舞弊", "不法", "違規", "違反誠信"],
    "重大資訊": ["重大資訊", "重大消息", "未公開資訊", "未公開"],
    "重大消息": ["重大資訊", "重大消息", "未公開資訊", "未公開"],
    "買股票": ["買股票", "買賣股票", "有價證券", "股票"],
    "關係人交易": ["關係人交易", "關係企業", "特定公司", "集團企業"],
    "衍生性商品": ["衍生性商品", "投機", "非避險", "非避險性交易", "避險"],
    "資產交易": ASSET_TERMS,
    "取得或處分資產": ASSET_TERMS,
    "不動產": ASSET_TERMS,
    "金融資產": ASSET_TERMS,
    "風險管理": ["風險管理"],
}


@dataclass(frozen=True)
class EvidenceAssessment:
    is_sufficient: bool
    note: str
    reasons: list[str]


def has_complete_citation(result: dict[str, Any]) -> bool:
    return all(result.get(field) not in (None, "") for field in REQUIRED_CITATION_FIELDS)


def expanded_query_terms(question: str) -> set[str]:
    terms = set(tokenize(question))
    for key, values in SYNONYMS.items():
        if key in question or any(value in question for value in values):
            terms.update(values)
    return {term for term in terms if term}


def query_overlap(question: str, result: dict[str, Any]) -> int:
    evidence_text = " ".join(
        str(result.get(field, ""))
        for field in ["policy_name", "article", "article_title", "text", "text_preview"]
    )
    terms = expanded_query_terms(question)
    return sum(1 for term in terms if term and term in evidence_text)


def boilerplate_ratio(results: list[dict[str, Any]]) -> float:
    if not results:
        return 1.0
    boilerplate_count = sum(1 for result in results if float(result.get("boilerplate_penalty", 0.0) or 0.0) > 0)
    return boilerplate_count / len(results)


def _has_text(result: dict[str, Any]) -> bool:
    return bool(str(result.get("text") or result.get("text_preview") or "").strip())


def _is_asset_query(question: str) -> bool:
    return any(term in question for term in ASSET_TERMS)


def _is_asset_policy_evidence(result: dict[str, Any]) -> bool:
    return result.get("document_id") == ASSET_POLICY_ID or result.get("policy_name") == ASSET_POLICY_NAME


def has_asset_policy_signal(question: str, results: list[dict[str, Any]]) -> bool:
    if not _is_asset_query(question):
        return False
    return any(
        _is_asset_policy_evidence(result) and (has_complete_citation(result) or _has_text(result))
        for result in results[:5]
    )


def assess_evidence(question: str, results: list[dict[str, Any]]) -> EvidenceAssessment:
    reasons: list[str] = []
    if not results:
        return EvidenceAssessment(False, "The retrieved evidence is insufficient to answer this confidently.", ["No evidence chunks were retrieved."])

    cited_results = [result for result in results if has_complete_citation(result)]
    if not cited_results:
        return EvidenceAssessment(False, "The retrieved evidence is insufficient to answer this confidently.", ["No retrieved chunk has complete citation metadata."])

    top_score = float(results[0].get("final_score", 0.0) or 0.0)
    if top_score < MIN_TOP_SCORE:
        reasons.append(f"Top final_score is low ({top_score:.2f}).")

    overlap_scores = [query_overlap(question, result) for result in cited_results[:5]]
    has_overlap = any(score > 0 for score in overlap_scores)
    if not has_overlap:
        reasons.append("Retrieved evidence has weak query-term or synonym overlap.")

    has_boost = any(
        float(result.get("keyword_boost", 0.0) or 0.0) > 0 or float(result.get("policy_boost", 0.0) or 0.0) > 0
        for result in cited_results[:5]
    )
    has_policy_family_signal = has_boost or has_asset_policy_signal(question, cited_results)
    if not has_policy_family_signal:
        reasons.append("No keyword or policy boost indicates a clearly relevant policy family.")

    ratio = boilerplate_ratio(cited_results[:5])
    if ratio >= BOILERPLATE_RATIO_LIMIT:
        reasons.append("Top evidence is mostly boilerplate or revision-history content.")

    if reasons:
        return EvidenceAssessment(False, "The retrieved evidence is insufficient to answer this confidently.", reasons)

    return EvidenceAssessment(True, "Evidence quality appears sufficient for a cautious policy-grounded demo answer.", ["Retrieved chunks include citations, relevant terms, and policy-family signals."])
