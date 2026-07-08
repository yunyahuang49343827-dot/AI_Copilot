"""Article-aware chunking for parsed WITS policy documents.

Day 3 scope only: convert parsed page-level JSON into article-aware JSONL
chunks. This module does not implement embeddings, retrieval, BM25, agents,
FastAPI, Streamlit, or evaluation.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "parsed_json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "chunks"
CHUNKS_FILE_NAME = "chunks.jsonl"
REPORT_FILE_NAME = "chunking_report.json"
DEFAULT_MAX_CHARS = 2000
MIN_SPLIT_CHARS = 2500

CHINESE_NUMERAL = "一二三四五六七八九十百千零〇兩0-9"
ARTICLE_RE = re.compile(
    rf"^\s*(?P<marker>第\s*[{CHINESE_NUMERAL}]+(?:\s*[{CHINESE_NUMERAL}]+)*\s*條(?:\s*之\s*[{CHINESE_NUMERAL}]+(?:\s*[{CHINESE_NUMERAL}]+)*)?)\s*(?P<punc>[：:、．.]?)\s*(?P<rest>.*)$"
)
CHAPTER_RE = re.compile(
    rf"^\s*(?P<marker>第\s*[{CHINESE_NUMERAL}]+\s*章)\s*(?P<title>.*)$"
)
SECTION_RE = re.compile(
    r"^\s*(?P<marker>[一二三四五六七八九十壹貳參肆伍陸柒捌玖拾]+[、.．])\s*(?P<title>.*)$"
)


@dataclass
class ArticleBlock:
    sequence: int
    document_id: str
    source_file: str
    file_name: str
    policy_name: str
    title: str
    chapter: str
    raw_article_marker: str
    article: str
    article_title: str
    page_start: int
    page_end: int
    lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip()


def normalize_marker(marker: str) -> str:
    """Normalize marker spacing and punctuation while preserving Chinese form."""

    normalized = re.sub(r"\s+", "", marker)
    return normalized.rstrip("：:、．.")


def clean_title_candidate(text: str) -> str:
    return text.strip().strip("：:、．. ")


def is_title_candidate(text: str) -> bool:
    candidate = clean_title_candidate(text)
    if not candidate:
        return False
    if len(candidate) > 24:
        return False
    if any(mark in candidate for mark in "。；;，,"):
        return False
    if ARTICLE_RE.match(candidate) or CHAPTER_RE.match(candidate):
        return False
    return True


def detect_article_marker(line: str) -> dict[str, str] | None:
    """Detect article markers only at the beginning of a line."""

    match = ARTICLE_RE.match(line)
    if not match:
        return None
    raw = f"{match.group('marker').rstrip()}{match.group('punc')}"
    rest = match.group("rest") or ""
    return {
        "raw_article_marker": raw,
        "article": normalize_marker(match.group("marker")),
        "rest": rest,
    }


def detect_chapter(line: str) -> str | None:
    match = CHAPTER_RE.match(line)
    if not match:
        return None
    raw_marker = normalize_marker(match.group("marker"))
    title = clean_title_candidate(match.group("title") or "")
    return f"{raw_marker} {title}".strip()


def detect_section_marker(line: str) -> dict[str, str] | None:
    """Detect fallback section markers only at the beginning of a line."""

    match = SECTION_RE.match(line)
    if not match:
        return None
    raw = match.group("marker")
    section_label = raw.rstrip("、.．")
    return {
        "raw_article_marker": raw,
        "article": f"section-{section_label}",
        "rest": match.group("title") or "",
    }


def iter_page_lines(document: dict[str, Any]) -> Iterable[tuple[int, str]]:
    for page in document.get("pages", []):
        page_number = int(page.get("page_number", 0))
        for line in (page.get("text") or "").splitlines():
            yield page_number, line.rstrip()


def make_block(
    sequence: int,
    document: dict[str, Any],
    chapter: str,
    marker: dict[str, str],
    page_number: int,
    first_line: str,
) -> ArticleBlock:
    title = clean_title_candidate(marker["rest"]) if is_title_candidate(marker["rest"]) else "TBD"
    return ArticleBlock(
        sequence=sequence,
        document_id=document.get("document_id", "TBD"),
        source_file=document.get("source_file", "TBD"),
        file_name=document.get("file_name", "TBD"),
        policy_name=document.get("policy_name", "TBD"),
        title=document.get("title") or document.get("policy_name") or "TBD",
        chapter=chapter or "TBD",
        raw_article_marker=marker["raw_article_marker"],
        article=marker["article"],
        article_title=title,
        page_start=page_number,
        page_end=page_number,
        lines=[first_line],
    )


def update_next_line_title(block: ArticleBlock, line: str) -> None:
    if block.article_title != "TBD":
        return
    if is_title_candidate(line):
        block.article_title = clean_title_candidate(line)


def article_blocks_for_document(document: dict[str, Any]) -> list[ArticleBlock]:
    blocks: list[ArticleBlock] = []
    current: ArticleBlock | None = None
    current_chapter = "TBD"
    sequence = 0

    for page_number, line in iter_page_lines(document):
        if not line.strip():
            if current is not None:
                current.lines.append(line)
            continue

        chapter = detect_chapter(line)
        if chapter:
            current_chapter = chapter
            if current is not None:
                current.lines.append(line)
                current.page_end = max(current.page_end, page_number)
            continue

        marker = detect_article_marker(line)
        if marker:
            if current is not None:
                blocks.append(current)
            sequence += 1
            current = make_block(sequence, document, current_chapter, marker, page_number, line)
            continue

        if current is not None:
            if len([value for value in current.lines if value.strip()]) == 1:
                update_next_line_title(current, line)
            current.lines.append(line)
            current.page_end = max(current.page_end, page_number)

    if current is not None:
        blocks.append(current)
    return blocks


def section_fallback_blocks_for_document(document: dict[str, Any]) -> list[ArticleBlock]:
    blocks: list[ArticleBlock] = []
    current: ArticleBlock | None = None
    sequence = 0

    for page_number, line in iter_page_lines(document):
        if not line.strip():
            if current is not None:
                current.lines.append(line)
            continue

        marker = detect_section_marker(line)
        if marker:
            if current is not None:
                blocks.append(current)
            sequence += 1
            current = make_block(sequence, document, "TBD", marker, page_number, line)
            continue

        if current is not None:
            current.lines.append(line)
            current.page_end = max(current.page_end, page_number)

    if current is not None:
        blocks.append(current)
    return blocks


def page_fallback_blocks_for_document(document: dict[str, Any]) -> list[ArticleBlock]:
    blocks: list[ArticleBlock] = []
    for index, page in enumerate(document.get("pages", []), start=1):
        text = (page.get("text") or "").strip()
        if not text:
            continue
        page_number = int(page.get("page_number", index))
        blocks.append(
            ArticleBlock(
                sequence=index,
                document_id=document.get("document_id", "TBD"),
                source_file=document.get("source_file", "TBD"),
                file_name=document.get("file_name", "TBD"),
                policy_name=document.get("policy_name", "TBD"),
                title=document.get("title") or document.get("policy_name") or "TBD",
                chapter="TBD",
                raw_article_marker="TBD",
                article=f"page-{page_number}",
                article_title="Fallback page chunk",
                page_start=page_number,
                page_end=page_number,
                lines=[text],
            )
        )
    return blocks


def split_text(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> list[str]:
    if len(text) <= MIN_SPLIT_CHARS:
        return [text]

    paragraphs = re.split(r"(\n\s*\n)", text)
    parts: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if not paragraph:
            continue
        if len(current) + len(paragraph) <= max_chars:
            current += paragraph
            continue
        if current.strip():
            parts.append(current.strip())
        if len(paragraph) <= max_chars:
            current = paragraph
        else:
            slices = [paragraph[i : i + max_chars] for i in range(0, len(paragraph), max_chars)]
            parts.extend(part.strip() for part in slices[:-1] if part.strip())
            current = slices[-1]
    if current.strip():
        parts.append(current.strip())
    return parts or [text]


def chunk_id_base(block: ArticleBlock, fallback_type: str | None = None) -> str:
    unit = "article"
    if fallback_type == "section":
        unit = "section"
    elif fallback_type == "page":
        unit = "page"
    return f"{block.document_id}-{unit}-{block.sequence:03d}"


def chunk_from_block(
    block: ArticleBlock,
    max_chars: int = DEFAULT_MAX_CHARS,
    fallback_type: str | None = None,
) -> list[dict[str, Any]]:
    text = block.text
    if not text:
        return []

    parent_id = chunk_id_base(block, fallback_type=fallback_type)
    text_parts = split_text(text, max_chars=max_chars)
    is_split = len(text_parts) > 1
    chunks: list[dict[str, Any]] = []

    for index, part in enumerate(text_parts, start=1):
        chunk_id = parent_id if not is_split else f"{parent_id}-part-{index:02d}"
        chunks.append(
            {
                "chunk_id": chunk_id,
                "document_id": block.document_id,
                "source_file": block.source_file,
                "file_name": block.file_name,
                "policy_name": block.policy_name,
                "title": block.title,
                "chapter": block.chapter,
                "raw_article_marker": block.raw_article_marker,
                "article": block.article,
                "article_title": block.article_title,
                "page_start": block.page_start,
                "page_end": block.page_end,
                "fallback_type": fallback_type,
                "part_index": index,
                "is_split_part": is_split,
                "parent_article_id": parent_id if is_split else None,
                "text": part,
                "char_count": len(part),
            }
        )
    return chunks


def load_documents(input_dir: Path) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("*.json")):
        if path.name == "parsing_report.json":
            continue
        docs.append(json.loads(path.read_text(encoding="utf-8")))
    return docs


def chunk_document(document: dict[str, Any], max_chars: int = DEFAULT_MAX_CHARS) -> tuple[list[dict[str, Any]], str | None]:
    blocks = article_blocks_for_document(document)
    fallback_type: str | None = None

    if not blocks:
        blocks = section_fallback_blocks_for_document(document)
        fallback_type = "section" if blocks else None

    if not blocks:
        blocks = page_fallback_blocks_for_document(document)
        fallback_type = "page" if blocks else None

    chunks: list[dict[str, Any]] = []
    for block in blocks:
        chunks.extend(chunk_from_block(block, max_chars=max_chars, fallback_type=fallback_type))
    return chunks, fallback_type


def build_report(
    documents: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    skipped_documents: list[dict[str, str]],
    fallback_documents: dict[str, list[str]],
) -> dict[str, Any]:
    char_counts = [int(chunk["char_count"]) for chunk in chunks]
    chunks_by_document: dict[str, int] = {}
    split_documents: dict[str, int] = {}
    for chunk in chunks:
        doc_id = chunk["document_id"]
        chunks_by_document[doc_id] = chunks_by_document.get(doc_id, 0) + 1
        if chunk["is_split_part"]:
            split_documents[doc_id] = split_documents.get(doc_id, 0) + 1

    return {
        "total_documents_seen": len(documents),
        "total_documents_chunked": len(chunks_by_document),
        "total_documents_skipped": len(skipped_documents),
        "total_chunks_created": len(chunks),
        "chunks_by_document": chunks_by_document,
        "fallback_documents": fallback_documents,
        "split_documents": split_documents,
        "skipped_documents": skipped_documents,
        "average_chunk_char_count": round(mean(char_counts), 2) if char_counts else 0,
        "max_chunk_char_count": max(char_counts) if char_counts else 0,
        "min_chunk_char_count": min(char_counts) if char_counts else 0,
        "examples": [
            {
                "chunk_id": chunk["chunk_id"],
                "policy_name": chunk["policy_name"],
                "chapter": chunk["chapter"],
                "article": chunk["article"],
                "article_title": chunk["article_title"],
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "char_count": chunk["char_count"],
                "text_preview": chunk["text"][:160],
            }
            for chunk in chunks[:5]
        ],
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_chunking(
    input_dir: Path = DEFAULT_INPUT_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> dict[str, Any]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Parsed JSON folder not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    documents = load_documents(input_dir)
    all_chunks: list[dict[str, Any]] = []
    skipped_documents: list[dict[str, str]] = []
    fallback_documents: dict[str, list[str]] = {"section_fallback": [], "page_fallback": []}

    for document in documents:
        document_id = document.get("document_id", "TBD")
        if document.get("extraction_status") != "success":
            skipped_documents.append(
                {
                    "document_id": document_id,
                    "file_name": document.get("file_name", "TBD"),
                    "reason": document.get("error_message", "extraction_status was not success"),
                }
            )
            continue
        chunks, fallback_type = chunk_document(document, max_chars=max_chars)
        if fallback_type == "section":
            fallback_documents["section_fallback"].append(document_id)
        elif fallback_type == "page":
            fallback_documents["page_fallback"].append(document_id)
        all_chunks.extend(chunks)

    chunks_path = output_dir / CHUNKS_FILE_NAME
    report_path = output_dir / REPORT_FILE_NAME
    write_jsonl(chunks_path, all_chunks)
    report = build_report(documents, all_chunks, skipped_documents, fallback_documents)
    write_json(report_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create article-aware chunks from parsed WITS policy JSON files.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_chunking(args.input_dir, args.output_dir, args.max_chars)
    print(
        "Chunking complete: "
        f"{report['total_chunks_created']} chunks, "
        f"{report['total_documents_chunked']} documents chunked, "
        f"{report['total_documents_skipped']} documents skipped."
    )
    fallback = report["fallback_documents"]
    print(
        "Fallback documents: "
        f"section={fallback.get('section_fallback', [])}, "
        f"page={fallback.get('page_fallback', [])}"
    )
    print(f"Output folder: {args.output_dir}")


if __name__ == "__main__":
    main()
