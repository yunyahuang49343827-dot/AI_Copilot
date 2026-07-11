"""Deterministic metrics for formal evaluation runs."""

from __future__ import annotations

import math
from typing import Any


def _normalize_list(values: Any) -> list[str]:
    """Normalize common scalar/list inputs into stripped non-empty strings."""
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple, set)):
        return []

    normalized: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            normalized.append(text)
    return normalized


def _dedupe_preserve_order(values: Any) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in _normalize_list(values):
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def _top_k(values: Any, k: int) -> list[str]:
    if k <= 0:
        return []
    return _dedupe_preserve_order(values)[:k]


def hit_at_k(retrieved_policy_ids: Any, expected_policy_ids: Any, k: int) -> float | None:
    expected = set(_normalize_list(expected_policy_ids))
    if not expected or k <= 0:
        return None if not expected else 0.0
    retrieved = _top_k(retrieved_policy_ids, k)
    return 1.0 if any(policy_id in expected for policy_id in retrieved) else 0.0


def precision_at_k(retrieved_policy_ids: Any, expected_policy_ids: Any, k: int) -> float | None:
    expected = set(_normalize_list(expected_policy_ids))
    if not expected or k <= 0:
        return None
    retrieved = _top_k(retrieved_policy_ids, k)
    relevant_count = sum(1 for policy_id in retrieved if policy_id in expected)
    return relevant_count / k


def recall_at_k(retrieved_policy_ids: Any, expected_policy_ids: Any, k: int) -> float | None:
    expected = set(_normalize_list(expected_policy_ids))
    if not expected or k <= 0:
        return None
    retrieved = _top_k(retrieved_policy_ids, k)
    relevant_count = len({policy_id for policy_id in retrieved if policy_id in expected})
    return relevant_count / len(expected)


def mrr(retrieved_policy_ids: Any, expected_policy_ids: Any) -> float | None:
    expected = set(_normalize_list(expected_policy_ids))
    if not expected:
        return None
    for index, policy_id in enumerate(_dedupe_preserve_order(retrieved_policy_ids), start=1):
        if policy_id in expected:
            return 1.0 / index
    return 0.0


def ndcg_at_k(retrieved_policy_ids: Any, expected_policy_ids: Any, k: int) -> float | None:
    expected = set(_normalize_list(expected_policy_ids))
    if not expected or k <= 0:
        return None

    retrieved = _top_k(retrieved_policy_ids, k)
    dcg = 0.0
    for rank, policy_id in enumerate(retrieved, start=1):
        relevance = 1.0 if policy_id in expected else 0.0
        dcg += relevance / math.log2(rank + 1)

    ideal_relevant_count = min(len(expected), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_relevant_count + 1))
    if idcg == 0:
        return None
    return dcg / idcg


def exact_match_any(actual_value: Any, expected_values: Any) -> float | None:
    expected = _normalize_list(expected_values)
    if not expected:
        return None
    if actual_value is None:
        actual = ""
    else:
        actual = str(actual_value).strip()
    return 1.0 if actual in expected else 0.0


def risk_accuracy(actual_risk: Any, expected_risk_list: Any) -> float | None:
    return exact_match_any(actual_risk, expected_risk_list)


def category_accuracy(actual_category: Any, expected_category_list: Any) -> float | None:
    return exact_match_any(actual_category, expected_category_list)


def insufficient_evidence_accuracy(actual_should_be_insufficient: Any, expected_should_be_insufficient: Any) -> float:
    return 1.0 if actual_should_be_insufficient == expected_should_be_insufficient else 0.0


def citation_coverage(answer_text: Any, citations: Any, should_be_insufficient: bool = False) -> float | None:
    if should_be_insufficient:
        return None
    if not isinstance(answer_text, str) or not answer_text.strip():
        return 0.0
    return 1.0 if _has_items(citations) else 0.0


def _has_items(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    return bool(value)


def checklist_presence_accuracy(actual_checklist_items: Any, should_be_insufficient: bool) -> float:
    has_checklist = _has_items(actual_checklist_items)
    if should_be_insufficient:
        return 0.0 if has_checklist else 1.0
    return 1.0 if has_checklist else 0.0
