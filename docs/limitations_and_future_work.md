# Limitations And Future Work

## Current Limitations

- Local prototype only.
- Not an official WITS system.
- Not legal advice.
- No authentication or API auth.
- No role-based access control.
- Runtime-only in-memory mock stores.
- No production database.
- Audit logs are runtime-only and not durable.
- No external workflow integrations.
- No email, Slack, Teams, Jira, ServiceNow, or real system mutation.
- Optional external DeepSeek API sends selected evidence to an external provider when enabled.
- Evaluation gold set is limited to 30 cases.
- Mixed-policy cross-document reasoning remains difficult.
- Reranker is experimental and default off.
- LLM layer is for Q&A answer wording only, not risk triage or action decisions.
- Source PDF redistribution rights must be checked before publishing PDFs.

## Future Work

- Authentication and API auth.
- Role-based access control.
- Persistent audit logs.
- Production database for agent runs, cases, and audit events.
- Cross-policy evidence planner.
- Safer mixed-policy workflow handling.
- Stronger reranker evaluation and regression thresholds.
- LangGraph interrupt/checkpointer version.
- Docker deployment.
- Cloud deployment.
- Admin document update and re-indexing workflow.
- LLM judge or groundedness evaluation for optional LLM mode.
- Human reviewer assignment and approval queues.
