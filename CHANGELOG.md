## Day 10 - Final Portfolio Documentation And Polish

### Added
- Rewrote README.md as an interview-ready portfolio front page with problem, solution, enterprise AI value, architecture overview, run commands, demo queries, safety disclaimer, limitations, and future work.
- Replaced docs/architecture.md with a current architecture explanation and Mermaid flow.
- Expanded docs/demo_script.md into a 3-5 minute demo talk track.
- Added docs/interview_story.md with STAR interview framing and role alignment.
- Added docs/limitations.md with honest safety, production, evaluation, and future-work notes.
- Added docs/api_examples.md with curl examples for local FastAPI endpoints.
- Added docs/screenshots_guide.md for portfolio screenshot planning and hygiene.

### Changed
- Polished Streamlit citation display to show source file basenames and replace TBD article titles with No explicit title in UI only.
- Added frontend helper tests for citation display cleanup.

### Not Added
- No retrieval scoring changes.
- No parsing, chunking, vector store, BM25, hybrid search, answer generation, risk triage, or backend schema changes.
- No database persistence, authentication, Docker, Kubernetes, cloud deployment, or external LLM API calls.

## Day 9 - Streamlit Demo UI And Evaluation Dashboard

### Added
- Added frontend/api_client.py for HTTP-only calls to the FastAPI backend with consistent ok/data/error results and graceful handling for connection errors, timeouts, non-200 responses, non-JSON responses, and missing fields.
- Added frontend/evaluation_examples.py with five deterministic portfolio smoke-test examples, multi-acceptable expected risks/categories, pass/fail logic, and pass-rate summary metrics.
- Added frontend/streamlit_app.py with Ask Copilot, Workflow Advice, Mock Cases, and Evaluation Dashboard tabs.
- Added session-state-backed UI results for QA, workflow advice, created case, case list, and evaluation results.
- Added tests/test_frontend_helpers.py for API client behavior and evaluation helper logic.
- Updated README.md with Day 9 backend/frontend run commands and local app URLs.

### Not Added
- No retrieval scoring changes.
- No parsing, chunking, vector store, BM25, hybrid search, answer generation, risk triage, or backend schema changes.
- No database persistence.
- No authentication.
- No Docker, Kubernetes, cloud deployment, or external LLM API calls.

## Day 8 - FastAPI Backend With In-Memory Mock Case Management

### Added
- Added backend/main.py with local FastAPI endpoints for health, grounded Q&A, workflow advice, case creation, case listing, and case lookup.
- Added backend/schemas.py with structured Pydantic request and response models, top_k validation, and blank-field validation.
- Added backend/services.py to convert existing retrieval, grounded Q&A, evidence checks, risk triage, and checklist logic into structured JSON responses.
- Added backend/mock_case_store.py for runtime-only in-memory mock case management. Cases disappear when the server restarts.
- Added tests/test_api.py with FastAPI TestClient coverage and monkeypatched service logic to avoid loading retrieval indexes in unit tests.
- Updated README.md with Day 8 run commands, curl examples, and local API docs URL.

### Not Added
- No Streamlit UI.
- No evaluation dashboard.
- No database persistence.
- No authentication.
- No Docker or Kubernetes.
- No external LLM API calls.
- No parsing, chunking, retrieval scoring, or index rebuild changes.

## Day 7 - Risk Triage And Workflow Checklist Advisor

### Added
- Added src/agent/risk_triage.py for deterministic evidence-based risk level and category classification.
- Added src/agent/checklist_builder.py for category-specific workflow checklist generation with citation labels on policy-supported items.
- Added src/agent/workflow_advisor.py CLI combining grounded Q&A, risk triage, checklist generation, citation table, evidence note, and demo disclaimer.
- Added category precedence for overlapping signals, prioritizing insider/material non-public information before other categories.
- Added demo triage wording: risk level is a demo triage priority based on retrieved policy evidence, not a legal determination.
- Added tests/test_risk_triage.py, tests/test_checklist_builder.py, and tests/test_workflow_advisor.py.
- Updated README.md with Day 7 run and test commands.

### Not Added
- No FastAPI backend.
- No Streamlit UI.
- No evaluation dashboard.
- No persistence or authentication.
- No external LLM API calls.
- No parsing, chunking, retrieval scoring, or index rebuild changes.

## Day 6.1 - Answer Quality Refinement

### Changed
- Added substantive evidence selection before answer generation so Direct Answer and Policy-Based Explanation focus on the strongest non-boilerplate chunks.
- Improved Direct Answer formatting to stay concise and avoid inserting long raw evidence text.
- Improved reporting-information questions to produce evidence-supported bullet lists.
- Improved derivative trading wording for speculative or non-hedging questions when trading-strategy evidence supports a restriction.
- Improved policy preference for insider-trading/material-information and related-party board-approval questions.
- Added tests for concise direct answers, bullet-list formatting, boilerplate exclusion from the main answer body, and derivative restriction wording.

### Not Added
- No retrieval scoring changes.
- No risk triage.
- No checklist generation.
- No agent workflow.
- No FastAPI or Streamlit implementation.
- No evaluation dashboard.

## Day 6 - Grounded Policy Q&A With Citations

### Added
- Added src/generation/answer_builder.py for local-first, template-based policy Q&A using existing hybrid retrieval.
- Added full chunk text hydration from data/chunks/chunks.jsonl by chunk_id when retrieval results include only previews.
- Added citation labels such as [C1] and a citation table mapping each label to policy name, article, article title, page span, source file, and chunk ID.
- Added guardrails for evidence sufficiency, citation completeness, query/synonym overlap, boost signals, and boilerplate-heavy evidence.
- Added the required portfolio demo disclaimer to every generated answer.
- Added disabled optional LLM adapter stub and prompts/policy_qa.md for future use without requiring external APIs.
- Added tests/test_answer_builder.py and tests/test_guardrails.py.
- Updated README.md with Day 6 CLI and test commands.

### Not Added
- No risk triage.
- No checklist generation.
- No full agent workflow.
- No FastAPI backend.
- No Streamlit UI.
- No evaluation dashboard.
- No external LLM API requirement.

## Day 5.1 - Boilerplate Penalty Refinement

### Added
- Added a small explainable `boilerplate_penalty` field to hybrid scoring.
- Penalized low-value revision/history and implementation boilerplate patterns such as `未盡事宜`, `修訂時亦同`, `第一次修訂`, and `其他事項`.
- Updated hybrid result formatting and report output to include the penalty.
- Added tests for boilerplate penalty detection, substantive no-penalty behavior, penalty subtraction, and formatting.

### Not Added
- No answer generation.
- No agent workflow.
- No FastAPI backend.
- No Streamlit UI.
- No evaluation dashboard.

## Day 5 - BM25 And Hybrid Retrieval

### Added
- Added `src/retrieval/bm25_store.py` for local BM25 keyword retrieval over `data/chunks/chunks.jsonl`.
- Added Chinese-friendly tokenization with compliance phrase preservation, Chinese bigrams, and English/alphanumeric terms.
- Added persisted BM25 index and `storage/bm25/bm25_index_report.json`.
- Added `src/retrieval/hybrid_search.py` to combine vector retrieval, BM25 retrieval, keyword boosts, policy boosts, and section-level derivative trading boosts.
- Added `src/retrieval/citations.py` for shared citation/result formatting.
- Added focused BM25 and hybrid retrieval tests that do not load the real embedding model.
- Updated README with Day 5 build and query commands.

### Not Added
- No cross-encoder reranking.
- No answer generation.
- No agent workflow.
- No FastAPI backend.
- No Streamlit UI.
- No evaluation dashboard.

## Day 4 - Local Chroma Vector Retrieval

### Added
- Added `src/config.py` for shared local paths, Chroma collection name, and configurable embedding model.
- Added `src/retrieval/embeddings.py` with an E5-compatible sentence-transformers wrapper using `passage:` and `query:` prefixes.
- Added `src/retrieval/vector_store.py` to build and query a local Chroma index from `data/chunks/chunks.jsonl`.
- Added clean collection recreation on build to avoid duplicate Chroma records.
- Added Chroma-safe citation metadata preservation, including `fallback_type`.
- Added `storage/chroma/vector_index_report.json` generation after successful builds.
- Added `tests/test_vector_store.py` for chunk loading, metadata safety, searchable text construction, and query result formatting without loading a real embedding model.
- Updated `README.md` with Day 4 build and query commands.

### Not Added
- No BM25.
- No hybrid retrieval.
- No reranking.
- No agent workflow.
- No FastAPI backend.
- No Streamlit UI.
- No evaluation dashboard.

## Day 3.1 - Section-Aware Fallback Chunking

### Added
- Added section-aware fallback chunking for documents without `第X條` article markers.
- Added support for line-start section markers such as `一、目的`, `二、適用範圍`, `壹、目的`, and `貳、適用範圍`.
- Added `fallback_type` to every chunk: `null`, `section`, or `page`.
- Updated fallback reporting to separate `section_fallback` and `page_fallback` documents.
- Added tests for section fallback and sentence-internal section references.

### Not Added
- No embeddings, retrieval, BM25, agent workflow, FastAPI, Streamlit, or evaluation.

## Day 3 - Article-Aware Chunking

### Added
- Added `src/retrieval/chunker.py` for article-aware chunking from parsed JSON files.
- Added support for compact and spaced Chinese article markers such as `第一條`, `第十條之一`, `第 一 條：`, and `第 十 二 條 之 一：`.
- Added line-start-only article detection to avoid splitting on article references inside normal sentences.
- Added chapter context preservation for markers such as `第一章` without using chapters as chunk units.
- Added raw and normalized article marker metadata on every chunk.
- Added recursive long-article splitting with `part_index`, `is_split_part`, and `parent_article_id` metadata.
- Added `data/chunks/chunks.jsonl` and `data/chunks/chunking_report.json` generation.
- Added `tests/test_article_chunker.py` for article marker, title detection, metadata, chapter, and split behavior.
- Updated `README.md` with Day 3 run and inspection commands.

### Not Added
- No embeddings.
- No vector retrieval.
- No BM25 or hybrid retrieval.
- No agent workflows.
- No FastAPI backend.
- No Streamlit UI.
- No evaluation implementation.

## Day 2 - PDF Parsing Pipeline

### Added
- Added src/ingestion/pdf_loader.py for manifest-aware, page-level PDF extraction with PyMuPDF.
- Added src/ingestion/parse_pdfs.py runnable with python -m src.ingestion.parse_pdfs.
- Added parser behavior that writes one per-document JSON file for every PDF, including failed parses.
- Added data/parsed_json/parsing_report.json generation with success/failure counts, output folder, per-file status, total pages, and total characters.
- Updated README.md with Day 2 run, dependency, and manual inspection commands.

### Not Added
- No article chunking.
- No embeddings or retrieval.
- No BM25 index.
- No agent workflows.
- No FastAPI backend.
- No Streamlit UI.
- No evaluation implementation.

### Notes
- The parser does not install PyMuPDF automatically. If fitz is unavailable, install dependencies with pip install -r requirements.txt or python -m pip install PyMuPDF.

## Day 1 - Repository Structure, Documentation, And Data Organization

### Added
- Created the approved project folder structure for data, storage, source modules, backend, frontend, evaluation, prompts, docs, and tests.
- Copied all 21 WITS public policy PDFs from `material/` into `data/raw_pdfs/`.
- Kept the original `material/` folder untouched.
- Created `data/documents_manifest.csv` with one row per PDF and the approved manifest columns.
- Added `README.md`, `docs/PROJECT_SCOPE.md`, `docs/business_value.md`, `docs/architecture.md`, and `docs/data_dictionary.md`.
- Added `requirements.txt`, `.env.example`, and `.gitignore`.

### Not Added
- No PDF parsing.
- No article chunking.
- No retrieval index.
- No Streamlit UI.
- No FastAPI backend.
- No evaluation implementation.
- No paid API or secret requirement.

### Manual Checks
- Confirm `data/raw_pdfs/` contains 21 PDFs.
- Confirm `data/documents_manifest.csv` has 21 data rows plus the header.
- Confirm `material/` still contains the original 21 PDFs.

### Next
- Day 2: implement PDF parsing into page-level JSON artifacts under `data/parsed_json/`.
