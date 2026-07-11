"""Template-based grounded policy Q&A for Day 6."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Callable

from src.config import CHUNKS_PATH
from src.generation.templates import citation_label, render_answer
from src.guardrails.evidence_checks import assess_evidence, has_complete_citation
from src.guardrails.safety import INSUFFICIENT_EVIDENCE_MESSAGE, cautious_prefix
from src.retrieval.citations import text_preview
from src.retrieval.hybrid_search import hybrid_search


RESTRICTION_TERMS = ["不得", "禁止", "應", "須", "不得洩露", "未公開", "十八小時", "非避險性交易"]
DERIVATIVE_RESTRICTION_TERMS = ["不得從事非避險性交易", "非避險性交易", "避險策略", "避險"]
BOARD_TERMS = ["董事會", "決議", "通過", "提交董事會", "提董事會"]
DATA_MARKERS = ["姓名", "身分證", "聯絡", "被檢舉", "具體事證", "檢舉內容", "證據"]
BOILERPLATE_PATTERNS = ["未盡事宜", "經董事會通過後", "公布實施", "修訂時亦同", "第一次修訂", "第二次修訂", "第三次修訂", "第四次修訂", "本辦法訂定於", "本程序訂定於", "本規範訂定於", "其他事項"]
INSIDER_POLICY_NAMES = ["防範內線交易管理程序", "內部重大資訊處理作業程序"]
ASSET_TRANSACTION_TERMS = ["資產", "併購", "合併", "分割", "收購", "處分"]


def load_full_chunks(chunks_path: Path = CHUNKS_PATH) -> dict[str, dict[str, Any]]:
    if not chunks_path.exists():
        return {}
    chunks: dict[str, dict[str, Any]] = {}
    with chunks_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            chunk = json.loads(line)
            chunks[str(chunk.get("chunk_id", ""))] = chunk
    return chunks


def hydrate_results(results: list[dict[str, Any]], chunks_by_id: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    chunks_by_id = chunks_by_id if chunks_by_id is not None else load_full_chunks()
    hydrated: list[dict[str, Any]] = []
    for result in results:
        merged = dict(result)
        full = chunks_by_id.get(str(result.get("chunk_id", "")))
        if full:
            for key, value in full.items():
                merged.setdefault(key, value)
            if full.get("text"):
                merged["text"] = full["text"]
        merged["text_preview"] = text_preview(merged.get("text", merged.get("text_preview", "")))
        hydrated.append(merged)
    return hydrated


def cited_results(results: list[dict[str, Any]], limit: int = 5) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for result in results:
        chunk_id = str(result.get("chunk_id", ""))
        if not chunk_id or chunk_id in seen or not has_complete_citation(result):
            continue
        seen.add(chunk_id)
        rows.append((citation_label(len(rows) + 1), result))
        if len(rows) >= limit:
            break
    return rows


def is_boilerplate_result(result: dict[str, Any]) -> bool:
    if float(result.get("boilerplate_penalty", 0.0) or 0.0) > 0:
        return True
    evidence_text = " ".join(str(result.get(field, "")) for field in ["article_title", "text", "text_preview"])
    return any(pattern in evidence_text for pattern in BOILERPLATE_PATTERNS)


def evidence_priority(question: str, row: tuple[str, dict[str, Any]]) -> tuple[int, float]:
    _, result = row
    policy_name = str(result.get("policy_name", ""))
    article = str(result.get("article", ""))
    article_title = str(result.get("article_title", ""))
    score = float(result.get("final_score", 0.0) or 0.0)
    priority = 0

    if is_boilerplate_result(result):
        priority -= 5
    if any(term in question for term in ["重大資訊", "重大消息", "未公開", "買股票", "買賣股票", "有價證券"]):
        if policy_name in INSIDER_POLICY_NAMES:
            priority += 4
        if "取得或處分資產處理程序" in policy_name and not any(term in question for term in ASSET_TRANSACTION_TERMS):
            priority -= 4
    if any(term in question for term in ["關係人交易", "關係企業", "董事會", "核准"]):
        if "關係人交易管理辦法" in policy_name:
            priority += 3
        if article in {"第十五條", "第十六條"}:
            priority += 3
    if any(term in question for term in ["投機", "非避險", "避險", "衍生性商品"]):
        if "從事衍生性商品交易處理規範" in policy_name:
            priority += 4
        if "交易策略" in article_title:
            priority += 4
    return priority, score


def select_body_evidence(question: str, citation_rows: list[tuple[str, dict[str, Any]]], limit: int = 3) -> list[tuple[str, dict[str, Any]]]:
    if not citation_rows:
        return []
    sorted_rows = sorted(citation_rows, key=lambda row: evidence_priority(question, row), reverse=True)
    substantive = [row for row in sorted_rows if not is_boilerplate_result(row[1])]
    if substantive:
        return substantive[:limit]
    return sorted_rows[:limit]


def compact_citation_labels(rows: list[tuple[str, dict[str, Any]]], limit: int = 2) -> str:
    return "".join(label for label, _ in rows[:limit])


def sentence_candidates(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[。；;])|\n", normalized)
    return [part.strip() for part in parts if part.strip()]


def first_relevant_sentence(result: dict[str, Any], terms: list[str] | None = None) -> str:
    text = result.get("text", "") or result.get("text_preview", "")
    candidates = sentence_candidates(text)
    if terms:
        for sentence in candidates:
            if any(term in sentence for term in terms):
                return sentence[:220]
    return (candidates[0] if candidates else text_preview(text))[:220]


def extract_required_information(results: list[dict[str, Any]]) -> list[str]:
    phrase_patterns = [
        (r"檢舉人之姓名、身分證號碼及可聯絡到檢舉人之聯絡方式", "檢舉人之姓名、身分證號碼及聯絡方式"),
        (r"被檢舉人之姓名或其他足茲識別被檢舉人身分特徵之資料", "被檢舉人之姓名或其他可識別身分特徵之資料"),
        (r"可供調查之具體事證", "可供調查之具體事證"),
        (r"檢舉內容", "檢舉內容"),
    ]
    items: list[str] = []
    for result in results:
        text = re.sub(r"\s+", " ", result.get("text", "") or result.get("text_preview", ""))
        for pattern, item in phrase_patterns:
            if re.search(pattern, text) and item not in items:
                items.append(item)
        for sentence in sentence_candidates(text):
            if len(sentence) > 120 or not any(marker in sentence for marker in DATA_MARKERS):
                continue
            cleaned = sentence.strip(" 0123456789.、")
            if cleaned and cleaned not in items:
                items.append(cleaned[:100])
            if len(items) >= 5:
                return items
    return items


def build_direct_answer(question: str, citation_rows: list[tuple[str, dict[str, Any]]], sufficient: bool) -> str:
    if not citation_rows or not sufficient:
        return f"{INSUFFICIENT_EVIDENCE_MESSAGE} Please use the citations below only as starting points for human review."

    labels = compact_citation_labels(citation_rows)
    joined_text = " ".join(result.get("text", "") for _, result in citation_rows[:3])

    if any(term in question for term in ["需要哪些資料", "哪些資料", "what information"]):
        items = extract_required_information([result for _, result in citation_rows])
        if items:
            bullets = "\n".join(f"- {item}" for item in items[:4])
            return f"{cautious_prefix()}, the retrieved policies identify the following evidence-supported information to provide {labels}:\n{bullets}"
        return f"{cautious_prefix()}, the retrieved evidence discusses reporting or handling requirements, but it does not clearly list all required information. {labels}"

    if "董事會" in question or "核准" in question:
        if any(term in joined_text for term in BOARD_TERMS):
            return f"{cautious_prefix()}, the retrieved related-party evidence indicates that board approval or board resolution may be required under specified conditions. Conditions may apply, so this should be reviewed by a human. {labels}"
        return f"{INSUFFICIENT_EVIDENCE_MESSAGE} The retrieved citations do not clearly establish a board-approval requirement."

    if any(term in question for term in ["投機", "非避險", "避險", "衍生性商品"]):
        if any(term in joined_text for term in DERIVATIVE_RESTRICTION_TERMS):
            return f"Based on the retrieved trading-strategy evidence, non-hedging or speculative derivative trading appears restricted. Human review is still required before applying this to a real transaction. {labels}"
        return f"{cautious_prefix()}, derivative trading appears tied to the retrieved trading-strategy and hedging evidence. Human review is required before any conclusion. {labels}"

    if any(term in question for term in ["可以", "是否可以", "can I", "買股票"]):
        if any(term in joined_text for term in RESTRICTION_TERMS):
            return f"{cautious_prefix()}, this appears restricted or not advisable without human review. The retrieved evidence discusses non-public/material information, confidentiality, or trading restrictions. {labels}"
        return f"{cautious_prefix()}, the evidence does not support a confident yes/no answer. Human review is required. {labels}"

    return f"{cautious_prefix()}, the retrieved policies provide relevant guidance, but the answer should be treated as a cautious summary rather than legal advice. {labels}"


def build_explanation(citation_rows: list[tuple[str, dict[str, Any]]], sufficient: bool) -> list[str]:
    if not citation_rows:
        return []
    if not sufficient:
        return [f"Retrieved evidence was weak; {label} may be related but should not be treated as a confident answer." for label, _ in citation_rows[:3]]
    lines: list[str] = []
    for label, result in citation_rows[:3]:
        sentence = first_relevant_sentence(result)
        lines.append(f"{label} {result.get('policy_name')} {result.get('article')} supports this point: {sentence}")
    return lines


def build_evidence_lines(citation_rows: list[tuple[str, dict[str, Any]]]) -> list[str]:
    return [
        f"{label} {first_relevant_sentence(result)}"
        for label, result in citation_rows[:5]
    ]


def build_grounded_answer(question: str, retrieved_results: list[dict[str, Any]]) -> str:
    hydrated = hydrate_results(retrieved_results)
    citation_rows = cited_results(hydrated)
    body_rows = select_body_evidence(question, citation_rows)
    assessment = assess_evidence(question, [result for _, result in body_rows or citation_rows])
    direct = build_direct_answer(question, body_rows, assessment.is_sufficient)
    explanation = build_explanation(body_rows, assessment.is_sufficient)
    evidence_lines = build_evidence_lines(body_rows)
    note = assessment.note
    if assessment.reasons:
        note += " Reasons: " + "; ".join(assessment.reasons)
    return render_answer(direct, explanation, evidence_lines, citation_rows, note)


def answer_question(question: str, top_k: int = 5, retriever: Callable[[str, int], list[dict[str, Any]]] = hybrid_search) -> str:
    results = retriever(question, top_k)
    return build_grounded_answer(question, results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grounded local policy Q&A using hybrid retrieval and citations.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(answer_question(args.query, top_k=args.top_k))


if __name__ == "__main__":
    main()
