import math

import pytest

from evaluation.metrics import (
    category_accuracy,
    checklist_presence_accuracy,
    citation_coverage,
    exact_match_any,
    hit_at_k,
    insufficient_evidence_accuracy,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    risk_accuracy,
)


def test_hit_at_k_success_and_failure():
    assert hit_at_k(["WITS-001", "WITS-018"], ["WITS-018"], 2) == 1.0
    assert hit_at_k(["WITS-001", "WITS-002"], ["WITS-018"], 2) == 0.0
    assert hit_at_k(["WITS-018"], ["WITS-018"], 0) == 0.0


def test_precision_at_k_divides_by_k():
    assert precision_at_k(["WITS-018"], ["WITS-018"], 5) == 0.2
    assert precision_at_k(["WITS-018", "WITS-009", "WITS-001"], ["WITS-018", "WITS-009"], 3) == pytest.approx(2 / 3)


def test_recall_at_k():
    assert recall_at_k(["WITS-018", "WITS-001"], ["WITS-018", "WITS-009"], 2) == 0.5
    assert recall_at_k(["WITS-018", "WITS-009"], ["WITS-018", "WITS-009"], 2) == 1.0


def test_mrr_rank_1_rank_3_and_missing():
    assert mrr(["WITS-018", "WITS-001"], ["WITS-018"]) == 1.0
    assert mrr(["WITS-001", "WITS-002", "WITS-018"], ["WITS-018"]) == pytest.approx(1 / 3)
    assert mrr(["WITS-001", "WITS-002"], ["WITS-018"]) == 0.0


def test_ndcg_at_k_perfect_and_imperfect_ranking():
    perfect = ndcg_at_k(["WITS-018", "WITS-009"], ["WITS-018", "WITS-009"], 2)
    imperfect = ndcg_at_k(["WITS-001", "WITS-018", "WITS-009"], ["WITS-018", "WITS-009"], 3)
    assert perfect == pytest.approx(1.0)
    assert imperfect is not None
    assert 0.0 < imperfect < 1.0


def test_empty_expected_policy_ids_return_none_for_retrieval_metrics():
    assert hit_at_k(["WITS-018"], [], 5) is None
    assert precision_at_k(["WITS-018"], [], 5) is None
    assert recall_at_k(["WITS-018"], [], 5) is None
    assert mrr(["WITS-018"], []) is None
    assert ndcg_at_k(["WITS-018"], [], 5) is None


def test_duplicate_retrieved_policy_ids_do_not_inflate_metrics():
    retrieved = ["WITS-001", "WITS-001", "WITS-018", "WITS-018"]
    expected = ["WITS-018"]

    assert precision_at_k(retrieved, expected, 3) == pytest.approx(1 / 3)
    assert recall_at_k(retrieved, expected, 3) == 1.0
    assert mrr(retrieved, expected) == 0.5
    assert ndcg_at_k(retrieved, expected, 3) == pytest.approx(1 / math.log2(3))


def test_exact_match_any_handles_list_string_none_and_whitespace():
    assert exact_match_any(" High ", ["Low", "High"]) == 1.0
    assert exact_match_any("High", "High") == 1.0
    assert exact_match_any(None, ["High"]) == 0.0
    assert exact_match_any("High", []) is None


def test_risk_accuracy_accepts_multiple_expected_labels():
    assert risk_accuracy("Medium", ["High", "Medium"]) == 1.0
    assert risk_accuracy("Low", ["High", "Medium"]) == 0.0


def test_category_accuracy_accepts_multiple_expected_labels():
    expected = ["Related-Party Transaction", "Board Approval / Governance Procedure"]
    assert category_accuracy(" Board Approval / Governance Procedure ", expected) == 1.0
    assert category_accuracy("Funds Lending", expected) == 0.0


def test_insufficient_evidence_accuracy():
    assert insufficient_evidence_accuracy(True, True) == 1.0
    assert insufficient_evidence_accuracy(False, False) == 1.0
    assert insufficient_evidence_accuracy(True, False) == 0.0


def test_citation_coverage_returns_1_when_answer_has_citations():
    assert citation_coverage("Based on evidence [C1]", [{"citation_id": "C1"}]) == 1.0


def test_citation_coverage_returns_0_when_answer_has_no_citations():
    assert citation_coverage("Based on evidence", []) == 0.0
    assert citation_coverage("", [{"citation_id": "C1"}]) == 0.0


def test_citation_coverage_returns_none_for_insufficient_evidence():
    assert citation_coverage("Insufficient evidence", [], should_be_insufficient=True) is None


def test_checklist_presence_accuracy_for_sufficient_and_insufficient_evidence():
    assert checklist_presence_accuracy(["Review policy"], should_be_insufficient=False) == 1.0
    assert checklist_presence_accuracy([], should_be_insufficient=False) == 0.0
    assert checklist_presence_accuracy(None, should_be_insufficient=True) == 1.0
    assert checklist_presence_accuracy([], should_be_insufficient=True) == 1.0
    assert checklist_presence_accuracy(["Review policy"], should_be_insufficient=True) == 0.0


def test_checklist_presence_accuracy_handles_non_empty_string():
    assert checklist_presence_accuracy("Review policy", should_be_insufficient=False) == 1.0
    assert checklist_presence_accuracy("Review policy", should_be_insufficient=True) == 0.0
    assert checklist_presence_accuracy("   ", should_be_insufficient=False) == 0.0


def test_functions_handle_none_and_empty_inputs_safely():
    assert precision_at_k(None, ["WITS-018"], 5) == 0.0
    assert recall_at_k(None, ["WITS-018"], 5) == 0.0
    assert mrr(None, ["WITS-018"]) == 0.0
    assert ndcg_at_k(None, ["WITS-018"], 5) == 0.0
    assert hit_at_k(None, ["WITS-018"], 5) == 0.0
    assert citation_coverage(None, None) == 0.0
