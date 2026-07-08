"""Small timing helpers for backend service-level performance metadata."""

from __future__ import annotations

import time


def start_timer() -> float:
    return time.perf_counter()


def elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)
