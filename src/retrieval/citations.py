"""Citation and result formatting helpers for retrieval outputs."""

from __future__ import annotations

from typing import Any


CITATION_FIELDS = [
    "chunk_id",
    "document_id",
    "policy_name",
    "article",
    "article_title",
    "page_start",
    "page_end",
    "source_file",
    "fallback_type",
]


def text_preview(text: str, limit: int = 220) -> str:
    return " ".join(str(text).split())[:limit]


def citation_metadata(chunk_or_metadata: dict[str, Any]) -> dict[str, Any]:
    return {field: chunk_or_metadata.get(field, "") for field in CITATION_FIELDS}


def format_retrieval_result(result: dict[str, Any], rank: int, score_label: str = "Score") -> str:
    page_start = result.get("page_start", "")
    page_end = result.get("page_end", "")
    score = result.get("score", result.get("final_score", result.get("bm25_score", "")))
    lines = [
        f"Rank: {rank}",
        f"{score_label}: {score}",
        f"Chunk ID: {result.get('chunk_id', '')}",
        f"Policy: {result.get('policy_name', '')}",
        f"Article: {result.get('article', '')}",
        f"Article Title: {result.get('article_title', '')}",
        f"Fallback Type: {result.get('fallback_type', '')}",
        f"Pages: {page_start}-{page_end}",
        f"Preview: {result.get('text_preview', text_preview(result.get('text', '')))}",
    ]
    return "\n".join(lines)
