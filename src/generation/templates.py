"""Template rendering for deterministic grounded policy Q&A."""

from __future__ import annotations

from typing import Any

from src.guardrails.disclaimers import DEMO_DISCLAIMER


def citation_label(index: int) -> str:
    return f"[C{index}]"


def citation_line(label: str, result: dict[str, Any]) -> str:
    pages = f"{result.get('page_start', '')}-{result.get('page_end', '')}"
    return (
        f"| {label} | {result.get('policy_name', '')} | {result.get('article', '')} | "
        f"{result.get('article_title', '')} | {pages} | {result.get('source_file', '')} | {result.get('chunk_id', '')} |"
    )


def render_answer(
    direct_answer: str,
    explanation: list[str],
    evidence_lines: list[str],
    citation_rows: list[tuple[str, dict[str, Any]]],
    evidence_note: str,
) -> str:
    explanation_text = "\n".join(f"- {line}" for line in explanation) if explanation else "- No supported explanation could be generated from the retrieved evidence."
    evidence_text = "\n".join(f"- {line}" for line in evidence_lines) if evidence_lines else "- No usable evidence excerpts were available."
    table_lines = [
        "| Citation ID | Policy Name | Article | Article Title | Pages | Source File | Chunk ID |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    table_lines.extend(citation_line(label, result) for label, result in citation_rows)

    return "\n\n".join(
        [
            "## Direct Answer\n" + direct_answer,
            "## Policy-Based Explanation\n" + explanation_text,
            "## Relevant Evidence\n" + evidence_text,
            "## Citation Table\n" + "\n".join(table_lines),
            "## Evidence Quality Note\n" + evidence_note,
            "## Demo Disclaimer\n" + DEMO_DISCLAIMER,
        ]
    )
