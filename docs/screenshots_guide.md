# Screenshots Guide

Use screenshots to show the working system without exposing local-only details or sensitive material.

## Ask Copilot Screenshot

- Demonstrates grounded Q&A with evidence quality and citations.
- Suggested caption: "Grounded policy answer with citation table."
- Avoid exposing full local file paths; use the Streamlit UI, which displays source file basenames.

## Workflow Advice Screenshot

- Demonstrates risk level, risk category, reasoning, and checklist.
- Suggested caption: "Risk triage and workflow checklist generated from retrieved policy evidence."
- Avoid presenting the output as legal advice.

## Mock Case Creation Screenshot

- Demonstrates FastAPI-backed in-memory case creation from workflow advice.
- Suggested caption: "Mock compliance case created from workflow advice."
- Use demo requester values such as `demo_user`; do not use real employee names.

## Evaluation Dashboard Screenshot

- Demonstrates the five deterministic smoke-test scenarios and pass rate.
- Suggested caption: "Lightweight demo evaluation dashboard for expected workflow outcomes."
- Mention that this is not a scientific benchmark.

## FastAPI Docs Screenshot

- Demonstrates structured local API endpoints.
- Suggested caption: "FastAPI endpoints for Q&A, workflow advice, and mock case management."
- Do not expose machine-specific paths or secrets.

## General Screenshot Hygiene

- Blur terminal usernames, local absolute paths, or private folders if visible.
- Do not show `.env` files or secrets.
- Confirm whether PDFs can be redistributed before showing document folders publicly.
- Keep captions clear that this is a local portfolio prototype, not an official WITS system.
