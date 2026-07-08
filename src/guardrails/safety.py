"""Safety helpers for local-first policy Q&A."""

INSUFFICIENT_EVIDENCE_MESSAGE = "The retrieved evidence is insufficient to answer this confidently."


def cautious_prefix() -> str:
    return "Based on the retrieved policy evidence"
