"""Isolated deterministic post-retrieval reranker for future experiments."""

from __future__ import annotations

from typing import Any

from src.retrieval.query_router import QueryRoute, classify_query_route


CANDIDATE_DOCUMENT_BOOST = 0.15
CANDIDATE_ARTICLE_BOOST = 0.08
MATCHED_TERM_BOOST = 0.05
OUTSIDE_CANDIDATE_DOCUMENTS_PENALTY = 0.05


def _numeric_score(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _result_haystack(result: dict[str, Any]) -> str:
    return " ".join(
        str(result.get(field, ""))
        for field in ["policy_name", "title", "article_title", "text_preview"]
    )


def _matches_route_terms(result: dict[str, Any], route: QueryRoute) -> bool:
    haystack = _result_haystack(result).lower()
    return any(term.lower() in haystack for term in route.matched_terms)


def _rerank_one(query: str, result: dict[str, Any], route: QueryRoute, original_rank: int) -> dict[str, Any]:
    reranked = dict(result)
    reranked["original_rank"] = original_rank
    score = _numeric_score(result.get("final_score"))
    reasons = ["baseline_score"]

    document_id = str(result.get("document_id", "") or "")
    article = str(result.get("article", "") or "")

    if document_id and document_id in route.candidate_documents:
        score += CANDIDATE_DOCUMENT_BOOST
        reasons.append("matched_candidate_document")

    if article and article in route.candidate_articles:
        score += CANDIDATE_ARTICLE_BOOST
        reasons.append("matched_candidate_article")

    if _matches_route_terms(result, route):
        score += MATCHED_TERM_BOOST
        reasons.append("matched_route_terms")

    if route.candidate_documents and document_id and document_id not in route.candidate_documents:
        score -= OUTSIDE_CANDIDATE_DOCUMENTS_PENALTY
        reasons.append("outside_candidate_documents_penalty")

    reranked["rerank_score"] = round(score, 6)
    reranked["rerank_reason"] = ", ".join(reasons)
    return reranked


def rerank_results(
    query: str,
    results: list[dict],
    route: QueryRoute | None = None,
) -> list[dict]:
    query_route = route if route is not None else classify_query_route(query)
    reranked = [
        _rerank_one(query, result, query_route, original_rank)
        for original_rank, result in enumerate(results, start=1)
    ]
    return sorted(reranked, key=lambda result: (-float(result["rerank_score"]), int(result["original_rank"])))
