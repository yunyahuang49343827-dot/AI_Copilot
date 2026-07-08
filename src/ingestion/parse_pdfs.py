"""Command-line entrypoint for Day 2 PDF parsing.

Run from the project root:
    python -m src.ingestion.parse_pdfs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.ingestion.pdf_loader import load_manifest, metadata_for_pdf, parse_pdf_safely


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw_pdfs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "parsed_json"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "documents_manifest.csv"
REPORT_FILE_NAME = "parsing_report.json"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def output_file_for(document_id: str, fallback_stem: str, output_dir: Path) -> Path:
    safe_id = document_id.strip() or fallback_stem
    safe_id = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in safe_id)
    return output_dir / f"{safe_id}.json"


def build_report(
    pdfs: list[Path],
    output_dir: Path,
    per_file_status: list[dict[str, Any]],
) -> dict[str, Any]:
    total_success = sum(1 for item in per_file_status if item["extraction_status"] == "success")
    total_pages = sum(int(item.get("total_pages", 0)) for item in per_file_status)
    total_chars = sum(int(item.get("characters_extracted", 0)) for item in per_file_status)
    total_failed = len(per_file_status) - total_success

    return {
        "total_pdfs_found": len(pdfs),
        "total_success": total_success,
        "total_failed": total_failed,
        "output_folder": str(output_dir),
        "per_file_status": per_file_status,
        "total_pages_extracted": total_pages,
        "total_characters_extracted": total_chars,
    }


def parse_all_pdfs(
    input_dir: Path = DEFAULT_INPUT_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> dict[str, Any]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input PDF folder not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(manifest_path)
    pdfs = sorted(input_dir.glob("*.pdf"))
    per_file_status: list[dict[str, Any]] = []

    for pdf_path in pdfs:
        metadata = metadata_for_pdf(pdf_path, manifest)
        parsed = parse_pdf_safely(pdf_path, metadata)
        output_path = output_file_for(metadata.document_id, pdf_path.stem, output_dir)
        write_json(output_path, parsed)

        pages = parsed.get("pages", [])
        characters = sum(int(page.get("char_count", 0)) for page in pages)
        status: dict[str, Any] = {
            "document_id": parsed["document_id"],
            "file_name": parsed["file_name"],
            "output_file": str(output_path),
            "extraction_status": parsed["extraction_status"],
            "total_pages": parsed.get("total_pages", 0),
            "characters_extracted": characters,
        }
        if parsed["extraction_status"] == "failed":
            status["error_message"] = parsed.get("error_message", "Unknown parsing error")
        per_file_status.append(status)

    report = build_report(pdfs, output_dir, per_file_status)
    write_json(output_dir / REPORT_FILE_NAME, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse WITS policy PDFs into page-level JSON files.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = parse_all_pdfs(args.input_dir, args.output_dir, args.manifest)
    print(
        "Parsed PDFs: "
        f"{report['total_success']} succeeded, "
        f"{report['total_failed']} failed, "
        f"{report['total_pdfs_found']} found."
    )
    print(f"Output folder: {report['output_folder']}")


if __name__ == "__main__":
    main()
