"""Local BM25 keyword retrieval for Day 5."""

from __future__ import annotations

import argparse
import json
import pickle
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from src.config import BM25_DIR, BM25_INDEX_PATH, BM25_INDEX_REPORT_PATH, CHUNKS_PATH
from src.retrieval.citations import format_retrieval_result, text_preview
from src.retrieval.vector_store import build_searchable_text, load_chunks


KNOWN_PHRASES = [
    "內線交易", "重大資訊", "重大消息", "未公開資訊", "未公開", "買股票", "買賣股票", "有價證券",
    "十八小時", "關係人交易", "關係企業", "特定公司", "集團企業", "資金貸與", "背書保證",
    "舉報", "檢舉", "通報", "申訴", "舞弊", "不法", "違規", "違反誠信",
    "衍生性商品", "投機", "非避險性交易", "非避險", "避險", "交易策略", "風險管理",
    "董事會", "審計委員會", "誠信經營", "道德行為",
]

ENGLISH_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+-]*")
CHINESE_RE = re.compile(r"[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    lowered = text.lower()

    for phrase in KNOWN_PHRASES:
        if phrase in text:
            tokens.append(phrase)

    tokens.extend(match.group(0).lower() for match in ENGLISH_RE.finditer(text))

    for match in CHINESE_RE.finditer(text):
        segment = match.group(0)
        if len(segment) == 1:
            tokens.append(segment)
            continue
        tokens.extend(segment[index : index + 2] for index in range(len(segment) - 1))

    # Useful for mixed queries like API; keep numeric/alphanumeric lowercase pass simple and transparent.
    tokens.extend(re.findall(r"[0-9]+", lowered))
    return tokens


def build_bm25_payload(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    documents = [build_searchable_text(chunk) for chunk in chunks]
    tokenized_corpus = [tokenize(document) for document in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    return {
        "chunks": chunks,
        "documents": documents,
        "tokenized_corpus": tokenized_corpus,
        "bm25": bm25,
    }


def save_payload(payload: dict[str, Any], path: Path = BM25_INDEX_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def load_payload(path: Path = BM25_INDEX_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"BM25 index not found: {path}. Run `python3 -m src.retrieval.bm25_store --build`."
        )
    with path.open("rb") as handle:
        return pickle.load(handle)


def write_report(total_chunks: int, path: Path = BM25_INDEX_REPORT_PATH) -> dict[str, Any]:
    report = {
        "total_chunks_loaded": total_chunks,
        "total_documents_indexed": total_chunks,
        "index_path": str(BM25_INDEX_PATH),
        "storage_path": str(BM25_DIR),
        "build_status": "success",
        "tokenizer": "known compliance phrases + Chinese bigrams + English/alphanumeric terms",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_index(chunks_path: Path = CHUNKS_PATH) -> dict[str, Any]:
    chunks = load_chunks(chunks_path)
    payload = build_bm25_payload(chunks)
    save_payload(payload)
    return write_report(len(chunks))


def search_payload(payload: dict[str, Any], query: str, top_k: int = 5) -> list[dict[str, Any]]:
    if not query.strip():
        raise ValueError("Query must not be empty.")
    scores = payload["bm25"].get_scores(tokenize(query))
    chunks = payload["chunks"]
    documents = payload["documents"]
    ranked_indices = sorted(range(len(scores)), key=lambda index: float(scores[index]), reverse=True)

    results: list[dict[str, Any]] = []
    for index in ranked_indices[:top_k]:
        chunk = dict(chunks[index])
        chunk["bm25_score"] = float(scores[index])
        chunk["score"] = float(scores[index])
        chunk["text_preview"] = text_preview(documents[index])
        results.append(chunk)
    return results


def query_index(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    return search_payload(load_payload(), query, top_k=top_k)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or query the local BM25 keyword index.")
    parser.add_argument("--build", action="store_true", help="Build BM25 from data/chunks/chunks.jsonl.")
    parser.add_argument("--query", type=str, help="Query the BM25 index.")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.build:
        report = build_index()
        print(
            "BM25 index build complete: "
            f"{report['total_documents_indexed']} documents indexed from {report['total_chunks_loaded']} chunks."
        )
        print(f"Report: {BM25_INDEX_REPORT_PATH}")
        return

    if args.query is not None:
        results = query_index(args.query, top_k=args.top_k)
        if not results:
            print("No results returned.")
            return
        print("\n\n".join(format_retrieval_result(result, rank, score_label="BM25 Score") for rank, result in enumerate(results, 1)))
        return

    raise SystemExit("Choose --build or --query.")


if __name__ == "__main__":
    main()
