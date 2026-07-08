"""PDF loading utilities for the Day 2 ingestion pipeline.

This module intentionally stops at page-level extraction. Article-aware chunking
belongs to Day 3 and later retrieval modules.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PYMUPDF_INSTALL_HELP = (
    "PyMuPDF is required for PDF parsing. Install project dependencies with "
    "pip install -r requirements.txt or install it directly with "
    "python -m pip install PyMuPDF."
)


@dataclass(frozen=True)
class DocumentMetadata:
    """Manifest metadata used to label parsed PDF output."""

    document_id: str
    file_name: str
    policy_name: str
    policy_family: str = "TBD"
    subdomain: str = "TBD"
    version: str = "TBD"
    effective_date: str = "TBD"
    language: str = "TBD"
    source_type: str = "TBD"

    @property
    def title(self) -> str:
        return self.policy_name if self.policy_name and self.policy_name != "TBD" else Path(self.file_name).stem


def load_manifest(manifest_path: Path) -> dict[str, DocumentMetadata]:
    """Load document metadata keyed by file name.

    Missing manifests are allowed so local parsing can still work from filenames.
    """

    if not manifest_path.exists():
        return {}

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        metadata: dict[str, DocumentMetadata] = {}
        for row in rows:
            file_name = (row.get("file_name") or "").strip()
            if not file_name:
                continue
            metadata[file_name] = DocumentMetadata(
                document_id=(row.get("doc_id") or Path(file_name).stem).strip(),
                file_name=file_name,
                policy_name=(row.get("policy_name") or Path(file_name).stem).strip(),
                policy_family=(row.get("policy_family") or "TBD").strip(),
                subdomain=(row.get("subdomain") or "TBD").strip(),
                version=(row.get("version") or "TBD").strip(),
                effective_date=(row.get("effective_date") or "TBD").strip(),
                language=(row.get("language") or "TBD").strip(),
                source_type=(row.get("source_type") or "TBD").strip(),
            )
        return metadata


def metadata_for_pdf(pdf_path: Path, manifest: dict[str, DocumentMetadata]) -> DocumentMetadata:
    """Return manifest metadata when available, otherwise a filename fallback."""

    found = manifest.get(pdf_path.name)
    if found:
        return found
    fallback_id = pdf_path.stem.replace(" ", "_")
    return DocumentMetadata(
        document_id=fallback_id,
        file_name=pdf_path.name,
        policy_name=pdf_path.stem,
    )


def build_failed_document(pdf_path: Path, metadata: DocumentMetadata, error_message: str) -> dict[str, Any]:
    """Build a per-document failure payload while preserving document identity."""

    return {
        "document_id": metadata.document_id,
        "source_file": str(pdf_path),
        "file_name": pdf_path.name,
        "title": metadata.title,
        "policy_name": metadata.policy_name,
        "policy_family": metadata.policy_family,
        "subdomain": metadata.subdomain,
        "version": metadata.version,
        "effective_date": metadata.effective_date,
        "language": metadata.language,
        "source_type": metadata.source_type,
        "total_pages": 0,
        "extraction_status": "failed",
        "error_message": error_message,
        "pages": [],
    }


def parse_pdf(pdf_path: Path, metadata: DocumentMetadata) -> dict[str, Any]:
    """Extract page-level text from a PDF with PyMuPDF."""

    try:
        import fitz  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError(PYMUPDF_INSTALL_HELP) from exc

    pages: list[dict[str, Any]] = []
    with fitz.open(pdf_path) as document:
        for index, page in enumerate(document, start=1):
            text = page.get_text("text") or ""
            pages.append(
                {
                    "page_number": index,
                    "text": text,
                    "char_count": len(text),
                }
            )

    return {
        "document_id": metadata.document_id,
        "source_file": str(pdf_path),
        "file_name": pdf_path.name,
        "title": metadata.title,
        "policy_name": metadata.policy_name,
        "policy_family": metadata.policy_family,
        "subdomain": metadata.subdomain,
        "version": metadata.version,
        "effective_date": metadata.effective_date,
        "language": metadata.language,
        "source_type": metadata.source_type,
        "total_pages": len(pages),
        "extraction_status": "success",
        "pages": pages,
    }


def parse_pdf_safely(pdf_path: Path, metadata: DocumentMetadata) -> dict[str, Any]:
    """Parse one PDF and convert any error into a failed document payload."""

    try:
        return parse_pdf(pdf_path, metadata)
    except Exception as exc:
        return build_failed_document(pdf_path, metadata, str(exc))
