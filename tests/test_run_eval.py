import json
from pathlib import Path

from evaluation.run_eval import (
    _default_retrieval_func,
    evaluate_one_example,
    extract_retrieved_policy_ids,
    main,
    mean_ignore_none,
    run_evaluation,
    write_json_report,
    write_markdown_report,
)


def _example(**overrides):
    example = {
        "id": "EVAL-T001",
        "query": "Can I trade before material news is public?",
        "expected_policy_ids": ["WITS-018"],
        "expected_policy_names": ["Insider Trading Policy"],
        "expected_risk": ["High"],
        "expected_category": ["Insider Trading / Material Non-Public Information"],
        "expected_answer_points": ["do not trade"],
        "should_be_insufficient": False,
        "notes": "test",
    }
    example.update(overrides)
    return example


def _retrieved_results():
    return [
        {"document_id": " WITS-001 ", "chunk_id": "c1"},
        {"document_id": "WITS-018", "chunk_id": "c2"},
        {"document_id": "WITS-018", "chunk_id": "c3"},
    ]


def _qa_response(**overrides):
    response = {
        "answer": "Use the cited policy evidence [C1].",
        "evidence_quality": "Sufficient evidence",
        "citations": [{"citation_id": "C1"}],
    }
    response.update(overrides)
    return response


def _workflow_response(**overrides):
    response = {
        "risk_level": "High",
        "risk_category": "Insider Trading / Material Non-Public Information",
        "workflow_checklist": ["Escalate for compliance review"],
        "citations": [{"citation_id": "C1"}],
    }
    response.update(overrides)
    return response


def test_extract_retrieved_policy_ids_dedupes_and_skips_empty_values():
    results = [
        {"document_id": " WITS-018 "},
        {"document_id": ""},
        {"document_id": None},
        {"document_id": "WITS-018"},
        {"document_id": "WITS-009"},
        {"chunk_id": "missing"},
        "not a dict",
    ]

    assert extract_retrieved_policy_ids(results) == ["WITS-018", "WITS-009"]


def test_mean_ignore_none_excludes_none_strings_and_bool():
    assert mean_ignore_none([1, None, "2", True, False, 3.0]) == 2.0
    assert mean_ignore_none([None, "1", True]) is None


def test_evaluate_one_example_success_uses_mocked_services_once_and_reuses_retrieval():
    calls = {"retrieval": 0, "qa": 0, "workflow": 0}

    def retrieval(query, top_k):
        calls["retrieval"] += 1
        assert query == "Can I trade before material news is public?"
        assert top_k == 5
        return _retrieved_results()

    def qa(query, top_k, retrieved_results):
        calls["qa"] += 1
        assert retrieved_results == _retrieved_results()
        return _qa_response()

    def workflow(query, top_k, retrieved_results):
        calls["workflow"] += 1
        assert retrieved_results == _retrieved_results()
        return _workflow_response()

    result = evaluate_one_example(_example(), retrieval_func=retrieval, qa_func=qa, workflow_func=workflow)

    assert result["status"] == "ok"
    assert result["error"] is None
    assert result["retrieved_policy_ids"] == ["WITS-001", "WITS-018"]
    assert result["actual_risk"] == "High"
    assert result["actual_category"] == "Insider Trading / Material Non-Public Information"
    assert result["citation_count"] == 1
    assert result["checklist_count"] == 1
    assert calls == {"retrieval": 1, "qa": 1, "workflow": 1}


def test_evaluate_one_example_computes_retrieval_metrics():
    result = evaluate_one_example(
        _example(),
        top_k=2,
        retrieval_func=lambda query, top_k: _retrieved_results(),
        qa_func=lambda query, top_k, retrieved_results: _qa_response(),
        workflow_func=lambda query, top_k, retrieved_results: _workflow_response(),
    )

    assert result["metrics"]["hit_at_k"] == 1.0
    assert result["metrics"]["hit_at_1"] == 0.0
    assert result["metrics"]["hit_at_3"] == 1.0
    assert result["metrics"]["hit_at_5"] == 1.0
    assert result["metrics"]["precision_at_k"] == 0.5
    assert result["metrics"]["precision_at_5"] == 0.2
    assert result["metrics"]["recall_at_k"] == 1.0
    assert result["metrics"]["recall_at_5"] == 1.0
    assert result["metrics"]["mrr"] == 0.5
    assert result["metrics"]["ndcg_at_k"] is not None
    assert result["metrics"]["ndcg_at_5"] is not None


def test_evaluate_one_example_computes_workflow_and_answer_metrics():
    result = evaluate_one_example(
        _example(),
        retrieval_func=lambda query, top_k: [{"document_id": "WITS-018"}],
        qa_func=lambda query, top_k, retrieved_results: _qa_response(),
        workflow_func=lambda query, top_k, retrieved_results: _workflow_response(),
    )

    assert result["metrics"]["risk_accuracy"] == 1.0
    assert result["metrics"]["category_accuracy"] == 1.0
    assert result["metrics"]["insufficient_evidence_accuracy"] == 1.0
    assert result["metrics"]["citation_coverage"] == 1.0
    assert result["metrics"]["checklist_presence_accuracy"] == 1.0


def test_evaluate_one_example_detects_insufficient_evidence_signals():
    example = _example(
        expected_policy_ids=[],
        expected_risk=["Insufficient Evidence"],
        expected_category=["Unknown or Insufficient Evidence"],
        should_be_insufficient=True,
    )

    result = evaluate_one_example(
        example,
        retrieval_func=lambda query, top_k: [],
        qa_func=lambda query, top_k, retrieved_results: _qa_response(
            answer="No policy answer.",
            evidence_quality="There is not enough evidence in the current corpus.",
            citations=[],
        ),
        workflow_func=lambda query, top_k, retrieved_results: _workflow_response(
            risk_level="Low",
            risk_category="General",
            workflow_checklist=[],
        ),
    )

    assert result["actual_should_be_insufficient"] is True
    assert result["metrics"]["insufficient_evidence_accuracy"] == 1.0
    assert result["metrics"]["citation_coverage"] is None
    assert result["metrics"]["checklist_presence_accuracy"] == 1.0


def test_evaluate_one_example_retrieval_exception_returns_error_status():
    def retrieval(query, top_k):
        raise RuntimeError("index unavailable")

    result = evaluate_one_example(
        _example(),
        retrieval_func=retrieval,
        qa_func=lambda query, top_k, retrieved_results: _qa_response(),
        workflow_func=lambda query, top_k, retrieved_results: _workflow_response(),
    )

    assert result["status"] == "error"
    assert result["error"] == "index unavailable"
    assert result["metrics"] == {}


def test_run_evaluation_aggregates_with_mocked_evaluator(tmp_path, monkeypatch):
    gold_path = tmp_path / "gold.jsonl"
    rows = [
        _example(id="EVAL-T001", expected_policy_ids=["WITS-018"]),
        _example(id="EVAL-T002", expected_policy_ids=["WITS-009"]),
    ]
    gold_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    def fake_evaluate(example, top_k=5, retrieval_func=None):
        if example["id"] == "EVAL-T001":
            return {
                "id": example["id"],
                "query": example["query"],
                "status": "ok",
                "error": None,
                "expected_policy_ids": example["expected_policy_ids"],
                "retrieved_policy_ids": ["WITS-018"],
                "metrics": {
                    "hit_at_k": 1.0,
                    "hit_at_1": 1.0,
                    "hit_at_3": 1.0,
                    "hit_at_5": 1.0,
                    "precision_at_k": 0.2,
                    "precision_at_5": 0.2,
                    "recall_at_k": 1.0,
                    "recall_at_5": 1.0,
                    "mrr": 1.0,
                    "ndcg_at_k": 1.0,
                    "ndcg_at_5": 1.0,
                    "risk_accuracy": 1.0,
                    "category_accuracy": 1.0,
                    "insufficient_evidence_accuracy": 1.0,
                    "citation_coverage": 1.0,
                    "checklist_presence_accuracy": 1.0,
                },
            }
        return {
            "id": example["id"],
            "query": example["query"],
            "status": "ok",
            "error": None,
            "expected_policy_ids": example["expected_policy_ids"],
            "retrieved_policy_ids": ["WITS-001"],
                "metrics": {
                    "hit_at_k": 0.0,
                    "hit_at_1": 0.0,
                    "hit_at_3": 0.0,
                    "hit_at_5": 0.0,
                    "precision_at_k": 0.0,
                    "precision_at_5": 0.0,
                    "recall_at_k": 0.0,
                    "recall_at_5": 0.0,
                    "mrr": 0.0,
                    "ndcg_at_k": 0.0,
                    "ndcg_at_5": 0.0,
                    "risk_accuracy": 0.0,
                "category_accuracy": 1.0,
                "insufficient_evidence_accuracy": 1.0,
                "citation_coverage": None,
                "checklist_presence_accuracy": 1.0,
            },
        }

    monkeypatch.setattr("evaluation.run_eval.evaluate_one_example", fake_evaluate)

    report = run_evaluation(gold_path, top_k=5)

    assert report["metadata"]["total_examples"] == 2
    assert list(report["retrieval_metrics"]) == [
        "hit_at_1",
        "hit_at_3",
        "hit_at_5",
        "precision_at_5",
        "recall_at_5",
        "mrr",
        "ndcg_at_5",
    ]
    assert list(report["workflow_metrics"]) == [
        "risk_accuracy",
        "category_accuracy",
        "insufficient_evidence_accuracy",
        "citation_coverage",
        "checklist_presence_accuracy",
    ]
    assert report["retrieval_metrics"]["hit_at_5"] == 0.5
    assert report["retrieval_metrics"]["precision_at_5"] == 0.1
    assert report["workflow_metrics"]["risk_accuracy"] == 0.5
    assert len(report["failed_examples"]) == 1
    assert report["failed_examples"][0]["id"] == "EVAL-T002"
    assert len(report["missing_gold_policy_examples"]) == 1
    assert report["missing_gold_policy_examples"][0]["id"] == "EVAL-T002"


def test_default_retrieval_func_passes_reranker_flags_to_hybrid_search(monkeypatch):
    calls = []

    def fake_hybrid_search(query, top_k=5, enable_reranker=False, rerank_candidate_k=None):
        calls.append(
            {
                "query": query,
                "top_k": top_k,
                "enable_reranker": enable_reranker,
                "rerank_candidate_k": rerank_candidate_k,
            }
        )
        return [{"document_id": "WITS-004"}]

    monkeypatch.setattr("src.retrieval.hybrid_search.hybrid_search", fake_hybrid_search)

    retrieval = _default_retrieval_func(enable_reranker=True, rerank_candidate_k=12)
    results = retrieval("資產交易", 5)

    assert results == [{"document_id": "WITS-004"}]
    assert calls == [
        {
            "query": "資產交易",
            "top_k": 5,
            "enable_reranker": True,
            "rerank_candidate_k": 12,
        }
    ]


def test_write_json_report_outputs_utf8_pretty_json(tmp_path):
    path = tmp_path / "report.json"
    report = {"metadata": {"total_examples": 1}, "message": "中文"}

    write_json_report(report, path)

    content = path.read_text(encoding="utf-8")
    assert '"message": "中文"' in content
    assert json.loads(content) == report


def test_write_markdown_report_outputs_expected_sections_without_full_answer_text(tmp_path):
    path = tmp_path / "report.md"
    report = {
        "metadata": {
            "gold_set_path": "evaluation/gold_set.jsonl",
            "top_k": 5,
            "total_examples": 1,
            "ok_examples": 1,
            "error_examples": 0,
        },
        "retrieval_metrics": {"hit_at_k": 1.0},
        "workflow_metrics": {"risk_accuracy": 1.0},
        "failed_examples": [],
        "missing_gold_policy_examples": [],
        "per_example_results": [
            {
                "id": "EVAL-T001",
                "status": "ok",
                "retrieved_policy_ids": ["WITS-018"],
                "actual_risk": "High",
                "actual_category": "Insider Trading",
                "metrics": {"hit_at_k": 1.0, "risk_accuracy": 1.0},
                "answer": "full answer should not appear",
            }
        ],
        "validation_warnings": [],
    }

    write_markdown_report(report, path)

    content = path.read_text(encoding="utf-8")
    assert "# v1.1 Formal Evaluation Report" in content
    assert "## Retrieval Metrics" in content
    assert "## Workflow Metrics" in content
    assert "## Per-example Results" in content
    assert "full answer should not appear" not in content


def test_cli_writes_default_report_names_to_output_dir(tmp_path, monkeypatch):
    fake_report = {
        "metadata": {
            "gold_set_path": "gold.jsonl",
            "top_k": 3,
            "total_examples": 0,
            "ok_examples": 0,
            "error_examples": 0,
        },
        "retrieval_metrics": {},
        "workflow_metrics": {},
        "per_example_results": [],
        "failed_examples": [],
        "missing_gold_policy_examples": [],
        "validation_warnings": [],
    }

    calls = {}

    def fake_run_evaluation(gold_set_path, top_k, enable_reranker=False, rerank_candidate_k=None):
        calls["gold_set_path"] = gold_set_path
        calls["top_k"] = top_k
        calls["enable_reranker"] = enable_reranker
        calls["rerank_candidate_k"] = rerank_candidate_k
        return fake_report

    monkeypatch.setattr("evaluation.run_eval.run_evaluation", fake_run_evaluation)

    exit_code = main(["--top-k", "3", "--gold-set", "gold.jsonl", "--output-dir", str(tmp_path)])

    assert exit_code == 0
    assert calls == {"gold_set_path": "gold.jsonl", "top_k": 3, "enable_reranker": False, "rerank_candidate_k": None}
    assert (tmp_path / "eval_report.json").exists()
    assert (tmp_path / "eval_report.md").exists()


def test_cli_forwards_reranker_flags_to_run_evaluation(tmp_path, monkeypatch):
    fake_report = {
        "metadata": {
            "gold_set_path": "gold.jsonl",
            "top_k": 5,
            "total_examples": 0,
            "ok_examples": 0,
            "error_examples": 0,
        },
        "retrieval_metrics": {},
        "workflow_metrics": {},
        "per_example_results": [],
        "failed_examples": [],
        "missing_gold_policy_examples": [],
        "validation_warnings": [],
    }
    calls = {}

    def fake_run_evaluation(gold_set_path, top_k, enable_reranker=False, rerank_candidate_k=None):
        calls["gold_set_path"] = gold_set_path
        calls["top_k"] = top_k
        calls["enable_reranker"] = enable_reranker
        calls["rerank_candidate_k"] = rerank_candidate_k
        return fake_report

    monkeypatch.setattr("evaluation.run_eval.run_evaluation", fake_run_evaluation)

    exit_code = main(
        [
            "--gold-set",
            "gold.jsonl",
            "--output-dir",
            str(tmp_path),
            "--enable-reranker",
            "--rerank-candidate-k",
            "12",
        ]
    )

    assert exit_code == 0
    assert calls == {"gold_set_path": "gold.jsonl", "top_k": 5, "enable_reranker": True, "rerank_candidate_k": 12}
