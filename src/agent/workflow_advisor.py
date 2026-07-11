"""CLI workflow advisor combining grounded Q&A, risk triage, and checklists."""

from __future__ import annotations

import argparse
from typing import Any, Callable

from src.agent.checklist_builder import build_checklist
from src.agent.risk_triage import RiskTriageResult, classify_risk
from src.generation.answer_builder import (
    build_direct_answer,
    build_evidence_lines,
    build_explanation,
    cited_results,
    hydrate_results,
    select_body_evidence,
)
from src.generation.templates import citation_line
from src.guardrails.disclaimers import DEMO_DISCLAIMER
from src.guardrails.evidence_checks import EvidenceAssessment, assess_evidence
from src.retrieval.hybrid_search import hybrid_search


DEMO_TRIAGE_NOTE = "Risk level is a demo triage priority based on retrieved policy evidence, not a legal determination."


def prepare_workflow_context(question: str, retrieved_results: list[dict[str, Any]]) -> dict[str, Any]:
    hydrated = hydrate_results(retrieved_results)
    citation_rows = cited_results(hydrated)
    body_rows = select_body_evidence(question, citation_rows)
    assessment = assess_evidence(question, [result for _, result in body_rows or citation_rows])
    direct_answer = build_direct_answer(question, body_rows, assessment.is_sufficient)
    explanation = build_explanation(body_rows, assessment.is_sufficient)
    evidence_lines = build_evidence_lines(body_rows)
    triage = classify_risk(question, body_rows or citation_rows, assessment)
    checklist = build_checklist(question, triage, body_rows or citation_rows)
    return {
        "citation_rows": citation_rows,
        "body_rows": body_rows,
        "assessment": assessment,
        "direct_answer": direct_answer,
        "explanation": explanation,
        "evidence_lines": evidence_lines,
        "triage": triage,
        "checklist": checklist,
    }


def render_grounded_answer_section(direct_answer: str, explanation: list[str], evidence_lines: list[str]) -> str:
    explanation_text = "\n".join(f"- {line}" for line in explanation) if explanation else "- No supported explanation could be generated from the retrieved evidence."
    evidence_text = "\n".join(f"- {line}" for line in evidence_lines) if evidence_lines else "- No usable evidence excerpts were available."
    return "\n".join([
        "Direct Answer:",
        direct_answer,
        "",
        "Policy-Based Explanation:",
        explanation_text,
        "",
        "Relevant Evidence:",
        evidence_text,
    ])


def render_risk_triage(triage: RiskTriageResult) -> str:
    return "\n".join([
        f"- Risk Level: {triage.risk_level}",
        f"- Risk Category: {triage.risk_category}",
        f"- Reasoning: {triage.reasoning}",
        f"- Demo Triage Note: {DEMO_TRIAGE_NOTE}",
    ])


def render_checklist(checklist: list[str]) -> str:
    return "\n".join(f"{index}. {step}" for index, step in enumerate(checklist, start=1))


def render_citation_table(citation_rows: list[tuple[str, dict[str, Any]]]) -> str:
    table_lines = [
        "| Citation ID | Policy Name | Article | Article Title | Pages | Source File | Chunk ID |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    table_lines.extend(citation_line(label, result) for label, result in citation_rows)
    return "\n".join(table_lines)


def evidence_note(assessment: EvidenceAssessment) -> str:
    note = assessment.note
    if assessment.reasons:
        note += " Reasons: " + "; ".join(assessment.reasons)
    return note


def render_workflow_advice(question: str, retrieved_results: list[dict[str, Any]]) -> str:
    context = prepare_workflow_context(question, retrieved_results)
    return "\n\n".join([
        "## Grounded Answer\n" + render_grounded_answer_section(context["direct_answer"], context["explanation"], context["evidence_lines"]),
        "## Risk Triage\n" + render_risk_triage(context["triage"]),
        "## Workflow Checklist\n" + render_checklist(context["checklist"]),
        "## Citation Table\n" + render_citation_table(context["citation_rows"]),
        "## Evidence Quality Note\n" + evidence_note(context["assessment"]),
        "## Demo Disclaimer\n" + DEMO_DISCLAIMER + " " + DEMO_TRIAGE_NOTE,
    ])


def advise_workflow(question: str, top_k: int = 5, retriever: Callable[[str, int], list[dict[str, Any]]] = hybrid_search) -> str:
    return render_workflow_advice(question, retriever(question, top_k))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local policy workflow advisor with grounded Q&A, risk triage, and checklist generation.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(advise_workflow(args.query, top_k=args.top_k))


if __name__ == "__main__":
    main()
