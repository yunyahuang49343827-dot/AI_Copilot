# Architecture

This project is a local-first enterprise policy copilot prototype. It turns policy PDFs into searchable evidence, grounded Q&A, deterministic workflow advice, controlled human-in-the-loop agent runs, mock case records, audit logs, and project-ready Streamlit/FastAPI demos.

## System Layers

```text
Policy PDFs
-> Parsing
-> Article-aware chunking
-> Vector index + BM25 index
-> Hybrid retrieval
-> Evidence selection and evidence quality checks
-> Deterministic grounded answer
-> Optional DeepSeek grounded answer layer
-> FastAPI response
-> Streamlit UI
```

```text
Policy Evidence
-> Workflow Advice
-> Risk Triage
-> Checklist Builder
-> Human-in-the-loop Agent
-> Approval Gate
-> Mock Case Store
-> Audit Log Store
```

## Retrieval Layer

The retrieval layer combines Chroma vector search with BM25 keyword retrieval. Vector search handles semantic similarity, while BM25 preserves exact policy terms such as `內線交易`, `關係人交易`, `資產`, `背書保證`, and `衍生性商品`. Hybrid search applies transparent policy boosts, keyword boosts, and boilerplate penalties.

The query router and reranker are implemented as isolated experimental retrieval utilities. The reranker is feature-flagged and default off, so production retrieval behavior remains unchanged unless explicitly enabled for diagnostics or evaluation experiments.

## Deterministic Answer Layer

The default Q&A path is deterministic. It hydrates retrieved chunks, selects citation rows, assesses evidence sufficiency, builds direct answers, explanations, evidence lines, citation summaries, disclaimers, and timing metadata. This default path is what the formal evaluation uses.

## Optional DeepSeek Grounded LLM Layer

The optional LLM layer is used only for Q&A answer wording when `use_llm=true`. It receives selected evidence rows, strict grounded prompt instructions, and the deterministic answer as fallback. It does not retrieve new documents, call tools, create cases, approve actions, alter risk level, alter workflow advice, alter audit logs, or route LangGraph.

Fallback occurs when evidence is insufficient, the LLM client is not configured, the provider fails, or the generated answer lacks a citation/reference to provided evidence. The API returns `generation_mode` and safe `llm_metadata` so the UI can show whether the answer is deterministic, LLM-grounded, or fallback.

## Workflow Layer

Workflow advice reuses the same policy evidence and adds deterministic risk triage, risk category selection, risk reasoning, and checklist generation. This path remains separate from the optional Q&A LLM layer.

## Human-In-The-Loop Agent Layer

The controlled agent workflow is service-level and deterministic:

- `src/agent/human_loop_agent.py`
- `src/agent/approval_gate.py`
- `src/agent/agent_tools.py`

The agent starts with workflow advice, evaluates the approval gate, prepares a pending action, and waits for human confirmation. High-risk cases require approval. Insufficient Evidence blocks case creation and recommends manual review. The only allowed action is mock case creation.

## FastAPI Layer

Main API paths:

- `POST /qa`
- `POST /workflow-advice`
- `POST /cases`
- `GET /cases`
- `POST /agent-runs`
- `GET /agent-runs`
- `GET /agent-runs/{run_id}`
- `POST /agent-runs/{run_id}/approve`
- `POST /agent-runs/{run_id}/reject`
- `GET /audit-logs`
- `GET /audit-logs/{audit_id}`

The stable `/agent-runs` flow remains the main operational path for the human-in-the-loop agent.

## Streamlit UI Layer

The Streamlit demo includes:

- Policy Q&A with citations, evidence quality, timing metadata, optional LLM checkbox, generation mode, and safe LLM metadata.
- Workflow Advice with risk triage and checklist.
- Compliance Cases for mock case workflows.
- Agent Workflow for start, inspect, approve, reject, and trace review.
- Audit Trail panel.
- Evaluation dashboard for lightweight demo examples.

## LangGraph Conditional Branch

LangGraph is implemented as a conditional human-review branch demo, not the default `/agent-runs` execution path.

```text
High risk or manual review case
-> LangGraph review branch
-> intake_node
-> workflow_node
-> approval_gate_node
-> route_by_decision
   |-> blocked_node
   |-> pending_approval_node
   |-> ready_for_confirmation_node
-> final_node
```

LangGraph does not create mock cases automatically and does not perform external actions.

## Audit Logging

Audit logging is runtime-only and in-memory. It records operational events such as agent run creation, approval required, approval granted, approval rejected, case created, and blocked insufficient evidence. Audit metadata is sanitized and should not contain full prompts, API keys, full evidence text, hidden reasoning, or timing metadata.

## Performance Metadata

Backend responses include timing metadata for retrieval, answer generation, workflow generation, agent run orchestration, store operations, audit operations, and total request time where applicable. The UI displays this as a performance visibility feature.

## Safety Boundaries

- Local-first by default.
- Optional DeepSeek API only when explicitly enabled for Q&A.
- No committed API keys or `.env` files.
- No external workflow mutation.
- No email, Slack, Teams, Jira, or ServiceNow calls.
- No automatic case creation from LangGraph.
- No LLM changes to risk, approval, audit, workflow, or case logic.
