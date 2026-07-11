# Project Scope

## In Scope

- Local-first Python portfolio demo over 21 WITS public policy PDFs.
- Repository structure for ingestion, retrieval, agent workflows, generation, guardrails, backend, frontend, evaluation, prompts, and docs.
- Bilingual demo style: English UI and technical framing, with Traditional Chinese policy names, article numbers, and citations preserved.
- Future article-aware chunking based on Chinese article markers.
- Future hybrid retrieval using Chroma and BM25.
- Future template-based grounded generation with citations.
- Future optional FastAPI mock case API that Streamlit can run without.

## Out Of Scope For Day 1

- PDF parsing
- Article chunking
- Vector indexing
- BM25 indexing
- Streamlit UI
- FastAPI backend
- Evaluation scripts
- LLM integration
- Cloud deployment
- Authentication or production integrations

## Explicit Non-Goals

- No paid service requirement by default.
- No LangChain or LangGraph in the first version.
- No Docker, Kubernetes, or cloud deployment unless requested later.
- No legal or compliance advice claims.
