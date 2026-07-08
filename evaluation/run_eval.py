"""Run the v1.1 formal evaluation against the local service layer."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from evaluation.gold_loader import DEFAULT_GOLD_SET_PATH, load_gold_set, validate_gold_set
from evaluation import metrics

DEFAULT_OUTPUT_DIR = Path("evaluation/reports")
DEFAULT_JSON_REPORT_PATH = DEFAULT_OUTPUT_DIR / "eval_report.json"
DEFAULT_MARKDOWN_REPORT_PATH = DEFAULT_OUTPUT_DIR / "eval_report.md"

RetrievalFunc = Callable[[str, int], list[dict[str, Any]]]
ResponseFunc = Callable[..., dict[str, Any]]


def extract_retrieved_policy_ids(retrieved_results: Any) -> list[str]:
    """Extract unique non-empty document ids from retrieval results."""
    if not isinstance(retrieved_results, list):
        return []

    seen: set[str] = set()
    policy_ids: list[str] = []
    for result in retrieved_results:
        if not isinstance(result, dict):
            continue
        document_id = result.get("document_id")
        if document_id is None:
            continue
        text = str(document_id).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        policy_ids.append(text)
    return policy_ids


def mean_ignore_none(values: Any) -> float | None:
    """Average numeric values while ignoring None, strings, and bools."""
    if not isinstance(values, list):
        return None

    numeric_values = [
        value
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if not numeric_values:
        return None
    return sum(numeric_values) / len(numeric_values)


def _default_retrieval_func(
    enable_reranker: bool = False,
    rerank_candidate_k: int | None = None,
) -> RetrievalFunc:
    from backend import services

    if not enable_reranker and rerank_candidate_k is None:
        return services._run_retrieval

    from src.retrieval.hybrid_search import hybrid_search

    def retrieve(query: str, top_k: int) -> list[dict[str, Any]]:
        return hybrid_search(
            query,
            top_k=top_k,
            enable_reranker=enable_reranker,
            rerank_candidate_k=rerank_candidate_k,
        )

    return retrieve


def _default_qa_func() -> ResponseFunc:
    from backend import services

    return services.build_qa_response


def _default_workflow_func() -> ResponseFunc:
    from backend import services

    return services.build_workflow_response


def _contains_insufficient_evidence_signal(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip().lower()
    return any(
        phrase in text
        for phrase in [
            "insufficient",
            "insufficient evidence",
            "no sufficient evidence",
            "not enough evidence",
        ]
    )


def _actual_should_be_insufficient(qa_response: dict[str, Any], workflow_response: dict[str, Any]) -> bool:
    return (
        workflow_response.get("risk_level") == "Insufficient Evidence"
        or workflow_response.get("risk_category") == "Unknown or Insufficient Evidence"
        or _contains_insufficient_evidence_signal(qa_response.get("evidence_quality"))
    )


def _count_items(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return 1 if value.strip() else 0
    if isinstance(value, (list, tuple, set, dict)):
        return len(value)
    return 1 if value else 0


def _metric_values(results: list[dict[str, Any]], key: str) -> list[Any]:
    return [result.get("metrics", {}).get(key) for result in results]


def _evaluate_metrics(
    *,
    answer_text: Any,
    citations: Any,
    checklist: Any,
    expected_policy_ids: Any,
    retrieved_policy_ids: Any,
    actual_risk: Any,
    expected_risk: Any,
    actual_category: Any,
    expected_category: Any,
    actual_should_be_insufficient: bool,
    expected_should_be_insufficient: bool,
    top_k: int,
) -> dict[str, float | None]:
    return {
        "hit_at_k": metrics.hit_at_k(retrieved_policy_ids, expected_policy_ids, top_k),
        "hit_at_1": metrics.hit_at_k(retrieved_policy_ids, expected_policy_ids, 1),
        "hit_at_3": metrics.hit_at_k(retrieved_policy_ids, expected_policy_ids, 3),
        "hit_at_5": metrics.hit_at_k(retrieved_policy_ids, expected_policy_ids, 5),
        "precision_at_k": metrics.precision_at_k(retrieved_policy_ids, expected_policy_ids, top_k),
        "precision_at_5": metrics.precision_at_k(retrieved_policy_ids, expected_policy_ids, 5),
        "recall_at_k": metrics.recall_at_k(retrieved_policy_ids, expected_policy_ids, top_k),
        "recall_at_5": metrics.recall_at_k(retrieved_policy_ids, expected_policy_ids, 5),
        "mrr": metrics.mrr(retrieved_policy_ids, expected_policy_ids),
        "ndcg_at_k": metrics.ndcg_at_k(retrieved_policy_ids, expected_policy_ids, top_k),
        "ndcg_at_5": metrics.ndcg_at_k(retrieved_policy_ids, expected_policy_ids, 5),
        "risk_accuracy": metrics.risk_accuracy(actual_risk, expected_risk),
        "category_accuracy": metrics.category_accuracy(actual_category, expected_category),
        "insufficient_evidence_accuracy": metrics.insufficient_evidence_accuracy(
            actual_should_be_insufficient,
            expected_should_be_insufficient,
        ),
        "citation_coverage": metrics.citation_coverage(
            answer_text,
            citations,
            should_be_insufficient=expected_should_be_insufficient,
        ),
        "checklist_presence_accuracy": metrics.checklist_presence_accuracy(
            checklist,
            should_be_insufficient=expected_should_be_insufficient,
        ),
    }


def evaluate_one_example(
    example: dict[str, Any],
    top_k: int = 5,
    retrieval_func: RetrievalFunc | None = None,
    qa_func: ResponseFunc | None = None,
    workflow_func: ResponseFunc | None = None,
) -> dict[str, Any]:
    """Evaluate one gold example, returning an error result on per-row failure."""
    example_id = example.get("id")
    query = example.get("query")
    expected_policy_ids = example.get("expected_policy_ids", [])
    expected_risk = example.get("expected_risk", [])
    expected_category = example.get("expected_category", [])
    expected_should_be_insufficient = bool(example.get("should_be_insufficient", False))

    base_result: dict[str, Any] = {
        "id": example_id,
        "query": query,
        "status": "ok",
        "error": None,
        "expected_policy_ids": expected_policy_ids,
        "retrieved_policy_ids": [],
        "retrieved_top_k": top_k,
        "expected_risk": expected_risk,
        "actual_risk": None,
        "expected_category": expected_category,
        "actual_category": None,
        "expected_should_be_insufficient": expected_should_be_insufficient,
        "actual_should_be_insufficient": None,
        "citation_count": 0,
        "checklist_count": 0,
        "metrics": {},
    }

    try:
        retrieval = retrieval_func or _default_retrieval_func()
        qa_builder = qa_func or _default_qa_func()
        workflow_builder = workflow_func or _default_workflow_func()

        retrieved_results = retrieval(str(query), top_k)
        retrieved_policy_ids = extract_retrieved_policy_ids(retrieved_results)
        qa_response = qa_builder(str(query), top_k=top_k, retrieved_results=retrieved_results)
        workflow_response = workflow_builder(str(query), top_k=top_k, retrieved_results=retrieved_results)

        actual_risk = workflow_response.get("risk_level")
        actual_category = workflow_response.get("risk_category")
        citations = qa_response.get("citations")
        checklist = workflow_response.get("workflow_checklist")
        actual_should_be_insufficient = _actual_should_be_insufficient(qa_response, workflow_response)

        base_result.update(
            {
                "retrieved_policy_ids": retrieved_policy_ids,
                "actual_risk": actual_risk,
                "actual_category": actual_category,
                "actual_should_be_insufficient": actual_should_be_insufficient,
                "citation_count": _count_items(citations),
                "checklist_count": _count_items(checklist),
                "metrics": _evaluate_metrics(
                    answer_text=qa_response.get("answer"),
                    citations=citations,
                    checklist=checklist,
                    expected_policy_ids=expected_policy_ids,
                    retrieved_policy_ids=retrieved_policy_ids,
                    actual_risk=actual_risk,
                    expected_risk=expected_risk,
                    actual_category=actual_category,
                    expected_category=expected_category,
                    actual_should_be_insufficient=actual_should_be_insufficient,
                    expected_should_be_insufficient=expected_should_be_insufficient,
                    top_k=top_k,
                ),
            }
        )
    except Exception as exc:
        base_result.update({"status": "error", "error": str(exc)})

    return base_result


def _is_failed_example(result: dict[str, Any]) -> bool:
    if result.get("status") != "ok":
        return True
    result_metrics = result.get("metrics", {})
    return any(
        result_metrics.get(key) == 0.0
        for key in [
            "risk_accuracy",
            "category_accuracy",
            "insufficient_evidence_accuracy",
        ]
    )


def _missing_gold_policy(result: dict[str, Any]) -> bool:
    expected = set(result.get("expected_policy_ids") or [])
    if not expected:
        return False
    retrieved = set(result.get("retrieved_policy_ids") or [])
    return expected.isdisjoint(retrieved)


def _summarize_example(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": result.get("id"),
        "query": result.get("query"),
        "status": result.get("status"),
        "error": result.get("error"),
        "expected_policy_ids": result.get("expected_policy_ids"),
        "retrieved_policy_ids": result.get("retrieved_policy_ids"),
        "expected_risk": result.get("expected_risk"),
        "actual_risk": result.get("actual_risk"),
        "expected_category": result.get("expected_category"),
        "actual_category": result.get("actual_category"),
        "metrics": result.get("metrics"),
    }


def run_evaluation(
    gold_set_path: str | Path = DEFAULT_GOLD_SET_PATH,
    top_k: int = 5,
    enable_reranker: bool = False,
    rerank_candidate_k: int | None = None,
) -> dict[str, Any]:
    """Run the formal evaluation and return a report dictionary."""
    examples = load_gold_set(gold_set_path)
    validation_warnings = validate_gold_set(examples)
    retrieval_func = _default_retrieval_func(
        enable_reranker=enable_reranker,
        rerank_candidate_k=rerank_candidate_k,
    )
    per_example_results = [
        evaluate_one_example(example, top_k=top_k, retrieval_func=retrieval_func)
        for example in examples
    ]

    retrieval_metric_keys = [
        "hit_at_1",
        "hit_at_3",
        "hit_at_5",
        "precision_at_5",
        "recall_at_5",
        "mrr",
        "ndcg_at_5",
    ]
    workflow_metric_keys = [
        "risk_accuracy",
        "category_accuracy",
        "insufficient_evidence_accuracy",
        "citation_coverage",
        "checklist_presence_accuracy",
    ]

    failed_examples = [_summarize_example(result) for result in per_example_results if _is_failed_example(result)]
    missing_gold_policy_examples = [
        _summarize_example(result) for result in per_example_results if _missing_gold_policy(result)
    ]

    return {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "gold_set_path": str(gold_set_path),
            "top_k": top_k,
            "enable_reranker": enable_reranker,
            "rerank_candidate_k": rerank_candidate_k,
            "total_examples": len(examples),
            "ok_examples": sum(1 for result in per_example_results if result.get("status") == "ok"),
            "error_examples": sum(1 for result in per_example_results if result.get("status") != "ok"),
        },
        "retrieval_metrics": {
            key: mean_ignore_none(_metric_values(per_example_results, key)) for key in retrieval_metric_keys
        },
        "workflow_metrics": {
            key: mean_ignore_none(_metric_values(per_example_results, key)) for key in workflow_metric_keys
        },
        "per_example_results": per_example_results,
        "failed_examples": failed_examples,
        "missing_gold_policy_examples": missing_gold_policy_examples,
        "validation_warnings": validation_warnings,
    }


def write_json_report(report: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _format_metric_table(metric_values: dict[str, Any]) -> list[str]:
    lines = ["| Metric | Value |", "| --- | ---: |"]
    for key, value in metric_values.items():
        display = "n/a" if value is None else f"{value:.4f}"
        lines.append(f"| {key} | {display} |")
    return lines


def write_markdown_report(report: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = report.get("metadata", {})
    lines = [
        "# v1.1 Formal Evaluation Report",
        "",
        "## Overall Summary",
        "",
        f"- Gold set path: `{metadata.get('gold_set_path')}`",
        f"- Top K: {metadata.get('top_k')}",
        f"- Total examples: {metadata.get('total_examples')}",
        f"- OK examples: {metadata.get('ok_examples')}",
        f"- Error examples: {metadata.get('error_examples')}",
        f"- Failed examples: {len(report.get('failed_examples', []))}",
        f"- Missing gold policy examples: {len(report.get('missing_gold_policy_examples', []))}",
        "",
        "## Retrieval Metrics",
        "",
        *_format_metric_table(report.get("retrieval_metrics", {})),
        "",
        "## Workflow Metrics",
        "",
        *_format_metric_table(report.get("workflow_metrics", {})),
        "",
        "## Failed Examples",
        "",
    ]

    failed_examples = report.get("failed_examples", [])
    if failed_examples:
        lines.extend(["| ID | Status | Error |", "| --- | --- | --- |"])
        for result in failed_examples:
            lines.append(f"| {result.get('id')} | {result.get('status')} | {result.get('error') or ''} |")
    else:
        lines.append("No failed examples.")

    lines.extend(["", "## Missing Gold Policy Examples", ""])
    missing_examples = report.get("missing_gold_policy_examples", [])
    if missing_examples:
        lines.extend(["| ID | Expected Policy IDs | Retrieved Policy IDs |", "| --- | --- | --- |"])
        for result in missing_examples:
            expected = ", ".join(result.get("expected_policy_ids") or [])
            retrieved = ", ".join(result.get("retrieved_policy_ids") or [])
            lines.append(f"| {result.get('id')} | {expected} | {retrieved} |")
    else:
        lines.append("No missing gold policy examples.")

    lines.extend(["", "## Per-example Results", ""])
    lines.extend(
        [
            "| ID | Status | Retrieved Policy IDs | Risk | Category | Key Metrics |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for result in report.get("per_example_results", []):
        result_metrics = result.get("metrics", {})
        retrieved = ", ".join(result.get("retrieved_policy_ids") or [])
        key_metrics = ", ".join(
            f"{key}={result_metrics.get(key)}"
            for key in ["hit_at_k", "risk_accuracy", "category_accuracy", "insufficient_evidence_accuracy"]
        )
        lines.append(
            "| {id} | {status} | {retrieved} | {risk} | {category} | {metrics} |".format(
                id=result.get("id"),
                status=result.get("status"),
                retrieved=retrieved,
                risk=result.get("actual_risk"),
                category=result.get("actual_category"),
                metrics=key_metrics,
            )
        )

    lines.extend(["", "## Notes / Limitations", ""])
    validation_warnings = report.get("validation_warnings", [])
    if validation_warnings:
        lines.append("- Gold set validation warnings were present in this run.")
    else:
        lines.append("- Gold set validation passed with no warnings.")
    lines.append("- This report excludes full answer text and full evidence text by design.")
    lines.append("- Metrics are deterministic and do not use external LLM judges.")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the v1.1 formal evaluation.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--gold-set", default=str(DEFAULT_GOLD_SET_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--enable-reranker", action="store_true", help="Run evaluation with the experimental retrieval reranker enabled.")
    parser.add_argument("--rerank-candidate-k", type=int, default=None, help="Number of merged candidates to rerank when reranker is enabled.")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    json_path = output_dir / "eval_report.json"
    markdown_path = output_dir / "eval_report.md"

    report = run_evaluation(
        gold_set_path=args.gold_set,
        top_k=args.top_k,
        enable_reranker=args.enable_reranker,
        rerank_candidate_k=args.rerank_candidate_k,
    )
    write_json_report(report, json_path)
    write_markdown_report(report, markdown_path)

    print(f"Wrote JSON report: {json_path}")
    print(f"Wrote Markdown report: {markdown_path}")
    print(f"Total examples: {report['metadata']['total_examples']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
