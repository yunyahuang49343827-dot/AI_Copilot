"""Hybrid retrieval combining vector search, BM25, and transparent boosts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb

from src.config import CHROMA_COLLECTION_NAME, CHROMA_DIR, HYBRID_DIR, HYBRID_SEARCH_REPORT_PATH
from src.retrieval.bm25_store import query_index as bm25_query_index
from src.retrieval.citations import text_preview
from src.retrieval.reranker import rerank_results
from src.retrieval.vector_store import query_index as vector_query_index


VECTOR_WEIGHT = 0.45
BM25_WEIGHT = 0.35
KEYWORD_BOOST_VALUE = 0.08
POLICY_BOOST_VALUE = 0.14
SECTION_POLICY_BOOST_VALUE = 0.10
SECTION_TITLE_BOOST_VALUE = 0.12
SECTION_TITLE_KEYWORD_BOOST_VALUE = 0.12
SECONDARY_POLICY_BOOST_VALUE = 0.06
BOILERPLATE_PENALTY_VALUE = 0.12
BOILERPLATE_PATTERNS = [
    "未盡事宜",
    "經董事會通過後",
    "公布實施",
    "公佈實施",
    "修訂時亦同",
    "第一次修訂",
    "第二次修訂",
    "第三次修訂",
    "第四次修訂",
    "本辦法訂定於",
    "本程序訂定於",
    "本規範訂定於",
    "其他事項",
]

BOOST_RULES = [
    {"terms": ["內線交易"], "policies": ["防範內線交易管理程序"]},
    {
        "terms": ["重大資訊", "重大消息", "未公開", "未公開資訊", "買股票", "買賣股票", "有價證券", "十八小時"],
        "policies": ["防範內線交易管理程序", "內部重大資訊處理作業程序"],
    },
    {
        "terms": ["舉報", "檢舉", "通報", "申訴", "舞弊", "不法", "違規", "違反誠信"],
        "policies": ["舉報制度", "道德行為準則", "誠信經營守則"],
    },
    {
        "terms": ["關係人交易", "關係企業", "特定公司", "集團企業"],
        "policies": ["與特定公司集團企業及關係人交易管理辦法", "取得或處分資產處理程序"],
    },
    {"terms": ["資金貸與"], "policies": ["資金貸與他人作業程序"]},
    {"terms": ["背書保證"], "policies": ["背書保證作業程序"]},
    {
        "terms": ["衍生性商品", "投機", "非避險性交易", "非避險", "避險"],
        "policies": ["從事衍生性商品交易處理規範"],
        "secondary_policies": ["取得或處分資產處理程序"],
        "section_title_contains": ["交易策略"],
    },
    {"terms": ["風險管理"], "policies": ["風險管理政策與程序"]},
]


def min_max_normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high == low:
        return [1.0 if high > 0 else 0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def matched_rules(query: str) -> list[dict[str, Any]]:
    return [rule for rule in BOOST_RULES if any(term in query for term in rule["terms"])]


def compute_boilerplate_penalty(result: dict[str, Any]) -> float:
    text = result.get("text", "") or result.get("document", "") or result.get("text_preview", "")
    article_title = result.get("article_title", "")
    combined = f"{article_title} {text}"
    if any(pattern in combined for pattern in BOILERPLATE_PATTERNS):
        return BOILERPLATE_PENALTY_VALUE
    return 0.0


def compute_boosts(query: str, result: dict[str, Any]) -> tuple[float, float]:
    keyword_boost = 0.0
    policy_boost = 0.0
    text = result.get("text", "") or result.get("document", "") or result.get("text_preview", "")
    policy_name = result.get("policy_name", "")
    article_title = result.get("article_title", "")

    for rule in matched_rules(query):
        if any(term in text or term in policy_name or term in article_title for term in rule["terms"]):
            keyword_boost += KEYWORD_BOOST_VALUE

        if policy_name in rule.get("policies", []):
            policy_boost += POLICY_BOOST_VALUE

        if policy_name in rule.get("secondary_policies", []):
            policy_boost += SECONDARY_POLICY_BOOST_VALUE

        if policy_name in rule.get("policies", []) and any(value in article_title for value in rule.get("section_title_contains", [])):
            policy_boost += SECTION_TITLE_BOOST_VALUE
            keyword_boost += SECTION_TITLE_KEYWORD_BOOST_VALUE

        if policy_name in rule.get("policies", []) and any(term in query for term in ["衍生性商品", "投機", "非避險性交易", "非避險", "避險"]):
            policy_boost += SECTION_POLICY_BOOST_VALUE

    return min(keyword_boost, 0.20), min(policy_boost, 0.25)


def vector_results_to_records(results: dict[str, Any]) -> list[dict[str, Any]]:
    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]
    records: list[dict[str, Any]] = []
    for index, metadata in enumerate(metadatas):
        record = dict(metadata)
        record["text"] = documents[index] if index < len(documents) else ""
        record["vector_distance"] = float(distances[index]) if index < len(distances) else None
        records.append(record)
    return records


def merge_results(vector_records: list[dict[str, Any]], bm25_records: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}

    vector_raw_scores = [1.0 / (1.0 + float(record["vector_distance"])) for record in vector_records if record.get("vector_distance") is not None]
    vector_norms = min_max_normalize(vector_raw_scores)
    for record, normalized in zip(vector_records, vector_norms):
        chunk_id = str(record["chunk_id"])
        merged[chunk_id] = {
            **record,
            "vector_score_normalized": normalized,
            "bm25_score": 0.0,
            "bm25_score_normalized": 0.0,
        }

    bm25_scores = [float(record.get("bm25_score", 0.0)) for record in bm25_records]
    bm25_norms = min_max_normalize(bm25_scores)
    for record, normalized in zip(bm25_records, bm25_norms):
        chunk_id = str(record["chunk_id"])
        existing = merged.setdefault(
            chunk_id,
            {
                **record,
                "vector_distance": None,
                "vector_score_normalized": 0.0,
            },
        )
        existing.update({key: value for key, value in record.items() if key not in {"score"}})
        existing["bm25_score"] = float(record.get("bm25_score", 0.0))
        existing["bm25_score_normalized"] = normalized

    for record in merged.values():
        keyword_boost, policy_boost = compute_boosts(query, record)
        boilerplate_penalty = compute_boilerplate_penalty(record)
        record["keyword_boost"] = keyword_boost
        record["policy_boost"] = policy_boost
        record["boilerplate_penalty"] = boilerplate_penalty
        record["final_score"] = (
            VECTOR_WEIGHT * float(record.get("vector_score_normalized", 0.0))
            + BM25_WEIGHT * float(record.get("bm25_score_normalized", 0.0))
            + keyword_boost
            + policy_boost
            - boilerplate_penalty
        )
        record["text_preview"] = text_preview(record.get("text", ""))

    return sorted(merged.values(), key=lambda item: item["final_score"], reverse=True)


def ensure_chroma_available() -> None:
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        client.get_collection(CHROMA_COLLECTION_NAME)
    except Exception as exc:
        raise FileNotFoundError(
            "Chroma vector index is missing or unavailable. Run `python3 -m src.retrieval.vector_store --build`."
        ) from exc


def hybrid_search(
    query: str,
    top_k: int = 5,
    enable_reranker: bool = False,
    rerank_candidate_k: int | None = None,
) -> list[dict[str, Any]]:
    candidate_k = max(rerank_candidate_k or top_k, top_k) if enable_reranker else top_k
    vector_top_k = max(candidate_k * 4, 20)
    bm25_top_k = max(candidate_k * 4, 20)
    ensure_chroma_available()
    vector_records = vector_results_to_records(vector_query_index(query, top_k=vector_top_k))
    bm25_records = bm25_query_index(query, top_k=bm25_top_k)
    merged = merge_results(vector_records, bm25_records, query)
    if not enable_reranker:
        return merged[:top_k]
    return rerank_results(query, merged[:candidate_k])[:top_k]


def write_search_report(query: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    HYBRID_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "query": query,
        "total_results_returned": len(results),
        "scoring": {
            "vector_weight": VECTOR_WEIGHT,
            "bm25_weight": BM25_WEIGHT,
            "keyword_boost_value": KEYWORD_BOOST_VALUE,
            "policy_boost_value": POLICY_BOOST_VALUE,
            "section_policy_boost_value": SECTION_POLICY_BOOST_VALUE,
            "section_title_boost_value": SECTION_TITLE_BOOST_VALUE,
            "section_title_keyword_boost_value": SECTION_TITLE_KEYWORD_BOOST_VALUE,
            "secondary_policy_boost_value": SECONDARY_POLICY_BOOST_VALUE,
            "boilerplate_penalty_value": BOILERPLATE_PENALTY_VALUE,
            "boilerplate_patterns": BOILERPLATE_PATTERNS,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "results": [
            {
                "rank": index,
                "chunk_id": result.get("chunk_id", ""),
                "policy_name": result.get("policy_name", ""),
                "article": result.get("article", ""),
                "article_title": result.get("article_title", ""),
                "final_score": result.get("final_score", 0.0),
                "boilerplate_penalty": result.get("boilerplate_penalty", 0.0),
            }
            for index, result in enumerate(results, start=1)
        ],
    }
    HYBRID_SEARCH_REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def format_hybrid_result(result: dict[str, Any], rank: int) -> str:
    return "\n".join(
        [
            f"Rank: {rank}",
            f"Final Score: {result.get('final_score', 0.0):.4f}",
            f"Vector Distance: {result.get('vector_distance', '')}",
            f"Vector Score Normalized: {result.get('vector_score_normalized', 0.0):.4f}",
            f"BM25 Score: {result.get('bm25_score', 0.0):.4f}",
            f"BM25 Score Normalized: {result.get('bm25_score_normalized', 0.0):.4f}",
            f"Keyword Boost: {result.get('keyword_boost', 0.0):.4f}",
            f"Policy Boost: {result.get('policy_boost', 0.0):.4f}",
            f"Boilerplate Penalty: {result.get('boilerplate_penalty', 0.0):.4f}",
            f"Chunk ID: {result.get('chunk_id', '')}",
            f"Document ID: {result.get('document_id', '')}",
            f"Policy: {result.get('policy_name', '')}",
            f"Article: {result.get('article', '')}",
            f"Article Title: {result.get('article_title', '')}",
            f"Fallback Type: {result.get('fallback_type', '')}",
            f"Pages: {result.get('page_start', '')}-{result.get('page_end', '')}",
            f"Source: {result.get('source_file', '')}",
            f"Preview: {result.get('text_preview', '')}",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid search over vector and BM25 indexes.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = hybrid_search(args.query, top_k=args.top_k)
    write_search_report(args.query, results)
    if not results:
        print("No results returned.")
        return
    print("\n\n".join(format_hybrid_result(result, rank) for rank, result in enumerate(results, start=1)))


if __name__ == "__main__":
    main()
