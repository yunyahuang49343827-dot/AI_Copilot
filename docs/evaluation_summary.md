# Evaluation Summary

## Purpose

The formal evaluation checks whether the copilot retrieves expected policy evidence and produces stable workflow outputs for a curated 30-case gold set. It is deterministic by default and does not depend on the optional DeepSeek LLM layer.

## Current Metrics

| Metric | Value |
| --- | ---: |
| Total examples | 30 |
| Error examples | 0 |
| Failed examples | 2 |
| Failed IDs | EVAL-015, EVAL-027 |
| hit_at_1 | 1.0000 |
| hit_at_3 | 1.0000 |
| hit_at_5 | 1.0000 |
| precision_at_5 | 0.2593 |
| recall_at_5 | 0.7284 |
| mrr | 1.0000 |
| ndcg_at_5 | 0.7864 |
| risk_accuracy | 0.9333 |
| category_accuracy | 0.9333 |
| insufficient_evidence_accuracy | 0.9333 |
| citation_coverage | 1.0000 |
| checklist_presence_accuracy | 0.9000 |

## Failed Cases

- `EVAL-015`
- `EVAL-027`

These remain hard because they involve mixed-policy or cross-policy evidence selection, where a single user query can touch asset procedure, related-party transaction governance, board approval, or other overlapping policy concepts.

## Reranker Comparison

The query router and reranker are implemented but remain experimental.

- Default-off reranker produced the same formal metrics as baseline.
- Larger candidate expansion retrieved more relevant documents in diagnostics.
- Candidate expansion also introduced an EVAL-026 risk regression.
- Conclusion: keep reranker feature-flagged and default off.

## DeepSeek LLM Note

The optional DeepSeek layer is an answer wording layer for Q&A only. It is not part of formal deterministic evaluation. It was manually verified through `/qa` and is exposed in the Streamlit Policy Q&A tab, but evaluation remains deterministic by default.

## Future Evaluation Work

- Add cross-policy evidence planner tests.
- Add larger mixed-policy gold cases.
- Evaluate reranker variants with regression thresholds.
- Add LLM groundedness checks when optional LLM mode is intentionally evaluated.
