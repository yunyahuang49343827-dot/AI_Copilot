"""Load and validate the formal evaluation gold set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_GOLD_SET_PATH = Path("evaluation/gold_set.jsonl")

REQUIRED_FIELDS = [
    "id",
    "query",
    "expected_policy_ids",
    "expected_policy_names",
    "expected_risk",
    "expected_category",
    "expected_answer_points",
    "should_be_insufficient",
    "notes",
]


def load_gold_set(path: str | Path = DEFAULT_GOLD_SET_PATH) -> list[dict[str, Any]]:
    """Load JSONL gold examples from disk.

    Raises ValueError with a readable line-level message if any row is not valid JSON.
    """
    gold_path = Path(path)
    examples: list[dict[str, Any]] = []
    with gold_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Invalid JSONL at line {line_number}: row must be a JSON object")
            examples.append(row)
    return examples


def _example_id(example: dict[str, Any]) -> str:
    value = example.get("id")
    return value if isinstance(value, str) and value else "<missing id>"


def validate_gold_example(example: dict[str, Any]) -> list[str]:
    """Return readable validation errors for one gold example."""
    errors: list[str] = []
    example_id = _example_id(example)

    for field in REQUIRED_FIELDS:
        if field not in example:
            errors.append(f"{example_id} missing required field: {field}")

    if "id" in example and not isinstance(example.get("id"), str):
        errors.append(f"{example_id} field id must be a string")
    elif isinstance(example.get("id"), str) and not example["id"].strip():
        errors.append(f"{example_id} field id must be non-empty")

    if "query" in example:
        query = example.get("query")
        if not isinstance(query, str) or not query.strip():
            errors.append(f"{example_id} field query must be a non-empty string")

    list_fields = [
        "expected_policy_ids",
        "expected_policy_names",
        "expected_risk",
        "expected_category",
        "expected_answer_points",
    ]
    for field in list_fields:
        if field in example and not isinstance(example.get(field), list):
            errors.append(f"{example_id} field {field} must be a list")

    for field in ["expected_risk", "expected_category"]:
        if field in example and isinstance(example.get(field), list) and not example[field]:
            errors.append(f"{example_id} field {field} must be a non-empty list")

    if "should_be_insufficient" in example and not isinstance(example.get("should_be_insufficient"), bool):
        errors.append(f"{example_id} field should_be_insufficient must be a bool")

    if example.get("should_be_insufficient") is True:
        policy_ids = example.get("expected_policy_ids")
        if isinstance(policy_ids, list) and policy_ids:
            errors.append(f"{example_id} insufficient example should have empty expected_policy_ids")

    return errors


def validate_gold_set(examples: list[dict[str, Any]]) -> list[str]:
    """Return readable validation errors for the full gold set."""
    errors: list[str] = []
    seen_ids: set[str] = set()

    for example in examples:
        errors.extend(validate_gold_example(example))
        example_id = example.get("id")
        if isinstance(example_id, str) and example_id.strip():
            if example_id in seen_ids:
                errors.append(f"Duplicate id found: {example_id}")
            seen_ids.add(example_id)

    return errors
