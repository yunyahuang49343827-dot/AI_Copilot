"""Local Chroma vector store for Day 4 semantic retrieval."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb

from src.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DIR,
    CHUNKS_PATH,
    EMBEDDING_MODEL,
    VECTOR_INDEX_REPORT_PATH,
)
from src.retrieval.embeddings import SentenceTransformerEmbedder


CITATION_METADATA_FIELDS = [
    "chunk_id",
    "document_id",
    "source_file",
    "file_name",
    "policy_name",
    "title",
    "chapter",
    "raw_article_marker",
    "article",
    "article_title",
    "fallback_type",
    "page_start",
    "page_end",
    "char_count",
]

EXAMPLE_QUERIES = [
    "內線交易",
    "重大資訊",
    "關係人交易",
    "資金貸與",
    "背書保證",
    "檢舉內部舞弊",
    "風險管理",
    "衍生性商品交易",
]


def load_chunks(chunks_path: Path = CHUNKS_PATH) -> list[dict[str, Any]]:
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}. Run Day 3 chunking first.")

    chunks: list[dict[str, Any]] = []
    with chunks_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {chunks_path}") from exc

    if not chunks:
        raise ValueError(f"No chunks loaded from {chunks_path}")
    return chunks


def build_searchable_text(chunk: dict[str, Any]) -> str:
    parts = [
        chunk.get("policy_name", ""),
        chunk.get("article", ""),
        chunk.get("article_title", ""),
        chunk.get("text", ""),
    ]
    return "\n".join(str(part) for part in parts if part is not None and str(part).strip())


def chroma_safe_value(value: Any) -> str | int | float | bool:
    if value is None:
        return ""
    if isinstance(value, (bool, int, float, str)):
        return value
    return json.dumps(value, ensure_ascii=False)


def metadata_for_chunk(chunk: dict[str, Any]) -> dict[str, str | int | float | bool]:
    return {field: chroma_safe_value(chunk.get(field, "")) for field in CITATION_METADATA_FIELDS}


def get_client(vector_store_path: Path = CHROMA_DIR) -> chromadb.PersistentClient:
    vector_store_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(vector_store_path))


def recreate_collection(client: chromadb.PersistentClient, collection_name: str = CHROMA_COLLECTION_NAME):
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    return client.create_collection(collection_name)


def write_report(
    total_chunks_loaded: int,
    total_vectors_indexed: int,
    embedding_model: str,
    vector_store_path: Path,
    collection_name: str,
    build_status: str,
) -> dict[str, Any]:
    report = {
        "total_chunks_loaded": total_chunks_loaded,
        "total_vectors_indexed": total_vectors_indexed,
        "embedding_model": embedding_model,
        "vector_store_path": str(vector_store_path),
        "collection_name": collection_name,
        "build_status": build_status,
        "example_queries": EXAMPLE_QUERIES,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    VECTOR_INDEX_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    VECTOR_INDEX_REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def build_index(
    chunks_path: Path = CHUNKS_PATH,
    vector_store_path: Path = CHROMA_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME,
    embedding_model: str = EMBEDDING_MODEL,
) -> dict[str, Any]:
    chunks = load_chunks(chunks_path)
    documents = [build_searchable_text(chunk) for chunk in chunks]
    ids = [str(chunk["chunk_id"]) for chunk in chunks]
    metadatas = [metadata_for_chunk(chunk) for chunk in chunks]

    embedder = SentenceTransformerEmbedder(embedding_model)
    embeddings = embedder.embed_texts(documents)

    client = get_client(vector_store_path)
    collection = recreate_collection(client, collection_name)
    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    report = write_report(
        total_chunks_loaded=len(chunks),
        total_vectors_indexed=len(ids),
        embedding_model=embedding_model,
        vector_store_path=vector_store_path,
        collection_name=collection_name,
        build_status="success",
    )
    return report


def get_collection(
    vector_store_path: Path = CHROMA_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME,
):
    client = get_client(vector_store_path)
    return client.get_collection(collection_name)


def query_index(
    query: str,
    top_k: int = 5,
    vector_store_path: Path = CHROMA_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME,
    embedding_model: str = EMBEDDING_MODEL,
) -> dict[str, Any]:
    if not query.strip():
        raise ValueError("Query must not be empty.")

    embedder = SentenceTransformerEmbedder(embedding_model)
    query_embedding = embedder.embed_query(query)
    collection = get_collection(vector_store_path, collection_name)
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )


def formatted_query_results(results: dict[str, Any]) -> list[str]:
    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    lines: list[str] = []
    for index, metadata in enumerate(metadatas, start=1):
        document = documents[index - 1] if index - 1 < len(documents) else ""
        distance = distances[index - 1] if index - 1 < len(distances) else ""
        preview = " ".join(str(document).split())[:220]
        page_start = metadata.get("page_start", "")
        page_end = metadata.get("page_end", "")
        fallback_type = metadata.get("fallback_type", "")
        lines.append(
            "\n".join(
                [
                    f"Rank: {index}",
                    f"Distance: {distance}",
                    f"Chunk ID: {metadata.get('chunk_id', '')}",
                    f"Policy: {metadata.get('policy_name', '')}",
                    f"Article: {metadata.get('article', '')}",
                    f"Article Title: {metadata.get('article_title', '')}",
                    f"Fallback Type: {fallback_type}",
                    f"Pages: {page_start}-{page_end}",
                    f"Preview: {preview}",
                ]
            )
        )
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or query the local WITS Chroma vector store.")
    parser.add_argument("--build", action="store_true", help="Build the Chroma vector index from chunks.jsonl.")
    parser.add_argument("--query", type=str, help="Query the Chroma vector index.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return.")
    parser.add_argument("--model", type=str, default=EMBEDDING_MODEL, help="Embedding model name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.build:
        report = build_index(embedding_model=args.model)
        print(
            "Vector index build complete: "
            f"{report['total_vectors_indexed']} vectors indexed from "
            f"{report['total_chunks_loaded']} chunks."
        )
        print(f"Vector store: {report['vector_store_path']}")
        print(f"Report: {VECTOR_INDEX_REPORT_PATH}")
        return

    if args.query is not None:
        results = query_index(args.query, top_k=args.top_k, embedding_model=args.model)
        formatted = formatted_query_results(results)
        if not formatted:
            print("No results returned.")
            return
        print("\n\n".join(formatted))
        return

    raise SystemExit("Choose --build or --query.")


if __name__ == "__main__":
    main()
