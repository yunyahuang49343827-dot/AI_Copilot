"""Shared local configuration for the WITS copilot demo."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
CHUNKS_PATH = DATA_DIR / "chunks" / "chunks.jsonl"

STORAGE_DIR = PROJECT_ROOT / "storage"
CHROMA_DIR = STORAGE_DIR / "chroma"
VECTOR_INDEX_REPORT_PATH = CHROMA_DIR / "vector_index_report.json"

DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

CHROMA_COLLECTION_NAME = "wits_policy_chunks"

BM25_DIR = STORAGE_DIR / "bm25"
BM25_INDEX_PATH = BM25_DIR / "bm25_index.pkl"
BM25_INDEX_REPORT_PATH = BM25_DIR / "bm25_index_report.json"

HYBRID_DIR = STORAGE_DIR / "hybrid"
HYBRID_SEARCH_REPORT_PATH = HYBRID_DIR / "hybrid_search_report.json"
