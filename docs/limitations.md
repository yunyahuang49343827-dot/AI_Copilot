# Limitations And Future Work

## Limitations

- This is not an official WITS system.
- This is not legal advice.
- This is not production-ready compliance software.
- It uses public or user-provided documents only.
- Source PDF redistribution rights must be checked before publishing PDFs to GitHub.
- The mock case store is in-memory only and disappears when the FastAPI server restarts.
- There is no authentication.
- There is no role-based access control.
- There is no production database.
- There is no durable audit trail.
- There is no external LLM reasoning by default.
- Template and rule-based generation is deterministic and explainable, but less flexible than an LLM-powered workflow.
- Some article titles may be `TBD` because PDF extraction and policy formatting vary.
- API responses may include local source file paths; the Streamlit UI displays basenames for portfolio readability.
- The evaluation dashboard is a deterministic smoke test, not a scientific benchmark.
- High-risk matters require qualified human review.

## Future Work

- Production database for durable case records and audit logs.
- Authentication and role-based access control.
- Human approval workflow with reviewer assignment and status tracking.
- Enterprise ticketing and collaboration integrations such as ServiceNow, Jira, Slack, or Teams.
- Docker and cloud deployment for controlled demo hosting.
- Optional LLM layer for richer synthesis while preserving citations and guardrails.
- More robust evaluation with a larger benchmark set and human-reviewed expected answers.
- Admin tools for document updates, re-indexing, and corpus versioning.
