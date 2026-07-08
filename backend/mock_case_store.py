"""Runtime-only in-memory mock case management for the FastAPI backend."""

from __future__ import annotations

from datetime import datetime, timezone
from random import choices
from string import ascii_uppercase, digits
from typing import Any

_CASES: dict[str, dict[str, Any]] = {}


def _new_case_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    suffix = "".join(choices(ascii_uppercase + digits, k=4))
    return f"CASE-{timestamp}-{suffix}"


def create_case(record: dict[str, Any]) -> dict[str, Any]:
    case_id = _new_case_id()
    while case_id in _CASES:
        case_id = _new_case_id()
    stored = {
        "case_id": case_id,
        "status": "Created",
        "created_at": datetime.now(timezone.utc).isoformat(),
        **record,
    }
    _CASES[case_id] = stored
    return stored


def get_case(case_id: str) -> dict[str, Any] | None:
    return _CASES.get(case_id)


def list_cases() -> list[dict[str, Any]]:
    return list(_CASES.values())


def clear_cases() -> None:
    _CASES.clear()
