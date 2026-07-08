# Project Demo Script

## Two-Minute Overview

"This is the WITS Governance, Compliance & Financial Control Copilot, a local-first enterprise AI prototype for policy-grounded compliance workflows. It parses internal policy PDFs, chunks them by article/section, uses hybrid retrieval with Chroma and BM25, returns grounded answers with citations, and adds deterministic risk triage and workflow checklists. The v2.0 layer adds controlled human-in-the-loop agent runs, FastAPI endpoints, Streamlit controls, audit logging, performance metadata, and a LangGraph conditional human-review branch. I also added an optional DeepSeek grounded answer layer for Q&A only; it is default off, evidence-constrained, and falls back to deterministic answers when evidence or configuration is insufficient."

## Five-Minute Demo Flow

1. Start FastAPI:

   ```bash
   python3 -m uvicorn backend.main:app --reload --port 8000
   ```

2. Start Streamlit:

   ```bash
   python3 -m streamlit run frontend/streamlit_app.py
   ```

3. Open Policy Q&A and ask:

   ```text
   公司還沒公告重大資訊，可以買股票嗎？
   ```

   Show the deterministic grounded answer, citations, evidence quality, disclaimer, and timing metadata.

4. Enable `Use LLM-assisted grounded answer` and ask the same question. Show `generation_mode=llm_grounded` if DeepSeek is configured, or fallback if not configured.

5. Ask the weak-evidence question:

   ```text
   這份文件有沒有說員工旅遊補助？
   ```

   Show insufficient evidence and explain why fallback is safer than invention.

6. Open Workflow Advice for the high-risk query. Show risk level, category, reasoning, checklist, citations, and evidence quality.

7. Open Agent Workflow. Start a high-risk agent run. Show PendingApproval and the pending action.

8. Approve one run to create a mock case. Reject another run to show the negative path.

9. Open Audit Trail and show events such as `AGENT_RUN_CREATED`, `APPROVAL_REQUIRED`, `APPROVAL_GRANTED`, `APPROVAL_REJECTED`, and `CASE_CREATED`.

10. Explain LangGraph as a conditional human-review branch, not the default execution path.

11. Explain formal evaluation results and remaining mixed-policy limitations.

## Ten-Minute Technical Walkthrough

1. **Ingestion and chunking**: PDFs are parsed with page metadata and split into article/section-aware chunks.
2. **Hybrid retrieval**: Chroma vector retrieval handles semantic similarity; BM25 preserves exact compliance terms.
3. **Evidence quality**: citation completeness, query overlap, policy/document signals, and boilerplate checks determine sufficiency.
4. **Deterministic Q&A**: default answers are template-based, grounded, cited, and used by formal evaluation.
5. **Optional DeepSeek layer**: `use_llm=true` only changes Q&A answer wording. It uses selected evidence and deterministic fallback.
6. **Workflow advice**: deterministic risk triage and checklist generation remain separate from the LLM.
7. **Human-in-the-loop agent**: high-risk actions require approval; insufficient evidence blocks case creation.
8. **FastAPI endpoints**: `/qa`, `/workflow-advice`, `/agent-runs`, `/cases`, and `/audit-logs`.
9. **Streamlit UI**: Q&A, workflow, cases, agent workflow, audit trail, performance metadata, and evaluation.
10. **LangGraph branch**: graph orchestration demo for conditional human review without automatic case creation.
11. **Evaluation**: 30-case gold set, deterministic metrics, two remaining mixed-policy failures.
12. **Future work**: RBAC, persistent audit logs, cross-policy evidence planner, stronger reranker evaluation, and production deployment.

## Query Walkthrough Notes

### 公司還沒公告重大資訊，可以買股票嗎？

This tests insider trading and material non-public information. The system should retrieve relevant policies, answer cautiously, mark this as high risk in workflow mode, and require human approval before mock case creation.

### 我想檢舉內部舞弊，應該提供哪些資料？

This tests whistleblowing and fraud reporting. The answer should list only evidence-supported information, such as reporter contact information, reported person information, report content, and concrete evidence.

### 衍生性商品交易可以投機嗎？

This tests derivatives and hedging controls. The system should find derivative trading policy evidence and highlight that speculative or non-hedging trading appears restricted.

### 關係人交易需要董事會核准嗎？

This tests related-party transaction governance. It is useful for showing workflow checklists and board-approval reasoning.

### 這份文件有沒有說員工旅遊補助？

This is the guardrail scenario. The system should not invent a policy answer. It should show insufficient evidence and recommend manual review.

## Closing

"The value here is the end-to-end enterprise AI solution pattern: document intelligence, hybrid retrieval, citations, deterministic guardrails, optional grounded LLM generation, human approval, auditability, API design, UI, LangGraph orchestration, timing metadata, and evaluation."
