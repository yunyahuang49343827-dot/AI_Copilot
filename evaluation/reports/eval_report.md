# v1.1 Formal Evaluation Report

## Overall Summary

- Gold set path: `evaluation/gold_set.jsonl`
- Top K: 5
- Total examples: 30
- OK examples: 30
- Error examples: 0
- Failed examples: 2
- Missing gold policy examples: 0

## Retrieval Metrics

| Metric | Value |
| --- | ---: |
| hit_at_1 | 1.0000 |
| hit_at_3 | 1.0000 |
| hit_at_5 | 1.0000 |
| precision_at_5 | 0.2593 |
| recall_at_5 | 0.7284 |
| mrr | 1.0000 |
| ndcg_at_5 | 0.7864 |

## Workflow Metrics

| Metric | Value |
| --- | ---: |
| risk_accuracy | 0.9333 |
| category_accuracy | 0.9333 |
| insufficient_evidence_accuracy | 0.9333 |
| citation_coverage | 1.0000 |
| checklist_presence_accuracy | 0.9000 |

## Failed Examples

| ID | Status | Error |
| --- | --- | --- |
| EVAL-015 | ok |  |
| EVAL-027 | ok |  |

## Missing Gold Policy Examples

No missing gold policy examples.

## Per-example Results

| ID | Status | Retrieved Policy IDs | Risk | Category | Key Metrics |
| --- | --- | --- | --- | --- | --- |
| EVAL-001 | ok | WITS-009, WITS-018, WITS-004 | High | Insider Trading / Material Non-Public Information | hit_at_k=1.0, risk_accuracy=1.0, category_accuracy=1.0, insufficient_evidence_accuracy=1.0 |
| EVAL-002 | ok | WITS-018 | High | Insider Trading / Material Non-Public Information | hit_at_k=1.0, risk_accuracy=1.0, category_accuracy=1.0, insufficient_evidence_accuracy=1.0 |
| EVAL-003 | ok | WITS-009, WITS-018, WITS-013 | High | Insider Trading / Material Non-Public Information | hit_at_k=1.0, risk_accuracy=1.0, category_accuracy=1.0, insufficient_evidence_accuracy=1.0 |
| EVAL-004 | ok | WITS-009 | High | Insider Trading / Material Non-Public Information | hit_at_k=1.0, risk_accuracy=1.0, category_accuracy=1.0, insufficient_evidence_accuracy=1.0 |
| EVAL-005 | ok | WITS-018 | High | Insider Trading / Material Non-Public Information | hit_at_k=1.0, risk_accuracy=1.0, category_accuracy=1.0, insufficient_evidence_accuracy=1.0 |
| EVAL-006 | ok | WITS-001, WITS-021, WITS-002, WITS-018 | High | Whistleblowing / Fraud Reporting | hit_at_k=1.0, risk_accuracy=1.0, category_accuracy=1.0, insufficient_evidence_accuracy=1.0 |
| EVAL-007 | ok | WITS-002, WITS-021 | High | Whistleblowing / Fraud Reporting | hit_at_k=1.0, risk_accuracy=1.0, category_accuracy=1.0, insufficient_evidence_accuracy=1.0 |
| EVAL-008 | ok | WITS-021, WITS-001, WITS-002, WITS-013 | High | Whistleblowing / Fraud Reporting | hit_at_k=1.0, risk_accuracy=1.0, category_accuracy=1.0, insufficient_evidence_accuracy=1.0 |
| EVAL-009 | ok | WITS-021, WITS-001 | High | Whistleblowing / Fraud Reporting | hit_at_k=1.0, risk_accuracy=1.0, category_accuracy=1.0, insufficient_evidence_accuracy=1.0 |
| EVAL-010 | ok | WITS-001, WITS-002, WITS-020, WITS-008 | High | Whistleblowing / Fraud Reporting | hit_at_k=1.0, risk_accuracy=1.0, category_accuracy=1.0, insufficient_evidence_accuracy=1.0 |
| EVAL-011 | ok | WITS-005 | High | Related-Party Transaction | hit_at_k=1.0, risk_accuracy=1.0, category_accuracy=1.0, insufficient_evidence_accuracy=1.0 |
| EVAL-012 | ok | WITS-005 | High | Related-Party Transaction | hit_at_k=1.0, risk_accuracy=1.0, category_accuracy=1.0, insufficient_evidence_accuracy=1.0 |
| EVAL-013 | ok | WITS-005, WITS-004 | High | Related-Party Transaction | hit_at_k=1.0, risk_accuracy=1.0, category_accuracy=1.0, insufficient_evidence_accuracy=1.0 |
| EVAL-014 | ok | WITS-005 | High | Related-Party Transaction | hit_at_k=1.0, risk_accuracy=1.0, category_accuracy=1.0, insufficient_evidence_accuracy=1.0 |
| EVAL-015 | ok | WITS-005, WITS-013 | Insufficient Evidence | Unknown or Insufficient Evidence | hit_at_k=1.0, risk_accuracy=0.0, category_accuracy=0.0, insufficient_evidence_accuracy=0.0 |
| EVAL-016 | ok | WITS-006, WITS-004 | High | Derivative Trading / Hedging Control | hit_at_k=1.0, risk_accuracy=1.0, category_accuracy=1.0, insufficient_evidence_accuracy=1.0 |
| EVAL-017 | ok | WITS-006, WITS-004 | Medium | Derivative Trading / Hedging Control | hit_at_k=1.0, risk_accuracy=1.0, category_accuracy=1.0, insufficient_evidence_accuracy=1.0 |
| EVAL-018 | ok | WITS-006, WITS-005, WITS-004 | High | Derivative Trading / Hedging Control | hit_at_k=1.0, risk_accuracy=1.0, category_accuracy=1.0, insufficient_evidence_accuracy=1.0 |
| EVAL-019 | ok | WITS-006, WITS-004 | Medium | Derivative Trading / Hedging Control | hit_at_k=1.0, risk_accuracy=1.0, category_accuracy=1.0, insufficient_evidence_accuracy=1.0 |
| EVAL-020 | ok | WITS-008 | High | Funds Lending | hit_at_k=1.0, risk_accuracy=1.0, category_accuracy=1.0, insufficient_evidence_accuracy=1.0 |
| EVAL-021 | ok | WITS-008 | High | Funds Lending | hit_at_k=1.0, risk_accuracy=1.0, category_accuracy=1.0, insufficient_evidence_accuracy=1.0 |
| EVAL-022 | ok | WITS-007 | High | Endorsement and Guarantee | hit_at_k=1.0, risk_accuracy=1.0, category_accuracy=1.0, insufficient_evidence_accuracy=1.0 |
| EVAL-023 | ok | WITS-007 | High | Endorsement and Guarantee | hit_at_k=1.0, risk_accuracy=1.0, category_accuracy=1.0, insufficient_evidence_accuracy=1.0 |
| EVAL-024 | ok | WITS-004 | High | Asset Acquisition or Disposal | hit_at_k=1.0, risk_accuracy=1.0, category_accuracy=1.0, insufficient_evidence_accuracy=1.0 |
| EVAL-025 | ok | WITS-004 | High | Asset Acquisition or Disposal | hit_at_k=1.0, risk_accuracy=1.0, category_accuracy=1.0, insufficient_evidence_accuracy=1.0 |
| EVAL-026 | ok | WITS-004 | Medium | Asset Acquisition or Disposal | hit_at_k=1.0, risk_accuracy=1.0, category_accuracy=1.0, insufficient_evidence_accuracy=1.0 |
| EVAL-027 | ok | WITS-005 | Insufficient Evidence | Unknown or Insufficient Evidence | hit_at_k=1.0, risk_accuracy=0.0, category_accuracy=0.0, insufficient_evidence_accuracy=0.0 |
| EVAL-028 | ok | WITS-013, WITS-020 | Insufficient Evidence | Unknown or Insufficient Evidence | hit_at_k=None, risk_accuracy=1.0, category_accuracy=1.0, insufficient_evidence_accuracy=1.0 |
| EVAL-029 | ok | WITS-020, WITS-004, WITS-003 | Insufficient Evidence | Unknown or Insufficient Evidence | hit_at_k=None, risk_accuracy=1.0, category_accuracy=1.0, insufficient_evidence_accuracy=1.0 |
| EVAL-030 | ok | WITS-004, WITS-008, WITS-013, WITS-020 | Insufficient Evidence | Unknown or Insufficient Evidence | hit_at_k=None, risk_accuracy=1.0, category_accuracy=1.0, insufficient_evidence_accuracy=1.0 |

## Notes / Limitations

- Gold set validation passed with no warnings.
- This report excludes full answer text and full evidence text by design.
- Metrics are deterministic and do not use external LLM judges.
