"""Optional LLM adapter stub. Disabled for the local-first Day 6 build."""

from __future__ import annotations


def is_enabled() -> bool:
    return False


def generate_with_llm(*_args, **_kwargs) -> str:
    raise RuntimeError("LLM generation is disabled by default. Day 6 uses deterministic templates only.")
