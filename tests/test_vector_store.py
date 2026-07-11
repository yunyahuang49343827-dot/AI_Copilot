import json
from pathlib import Path

from src.retrieval.vector_store import (
    build_searchable_text,
    formatted_query_results,
    load_chunks,
    metadata_for_chunk,
)


def sample_chunk():
    return {
        "chunk_id": "WITS-001-article-001",
        "document_id": "WITS-001",
        "source_file": "/tmp/source.pdf",
        "file_name": "source.pdf",
        "policy_name": "道德行為準則",
        "title": "道德行為準則",
        "chapter": "TBD",
        "raw_article_marker": "第一條",
        "article": "第一條",
        "article_title": "目的",
        "fallback_type": None,
        "page_start": 1,
        "page_end": 2,
        "text": "政策本文",
        "char_count": 4,
    }


def test_load_chunks_from_jsonl(tmp_path: Path):
    path = tmp_path / "chunks.jsonl"
    path.write_text(json.dumps(sample_chunk(), ensure_ascii=False) + "\n", encoding="utf-8")

    chunks = load_chunks(path)

    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == "WITS-001-article-001"


def test_metadata_fields_are_preserved_and_chroma_safe():
    metadata = metadata_for_chunk(sample_chunk())

    assert metadata["chunk_id"] == "WITS-001-article-001"
    assert metadata["policy_name"] == "道德行為準則"
    assert metadata["article"] == "第一條"
    assert metadata["fallback_type"] == ""
    assert metadata["page_start"] == 1
    assert metadata["page_end"] == 2


def test_searchable_text_includes_policy_context():
    searchable = build_searchable_text(sample_chunk())

    assert "道德行為準則" in searchable
    assert "第一條" in searchable
    assert "目的" in searchable
    assert "政策本文" in searchable


def test_result_formatting_does_not_crash():
    results = {
        "documents": [["道德行為準則\n第一條\n目的\n政策本文"]],
        "metadatas": [[metadata_for_chunk(sample_chunk())]],
        "distances": [[0.123]],
    }

    lines = formatted_query_results(results)

    assert len(lines) == 1
    assert "Rank: 1" in lines[0]
    assert "Distance: 0.123" in lines[0]
    assert "Chunk ID: WITS-001-article-001" in lines[0]
