# Data Dictionary

## `data/documents_manifest.csv`

| Column | Meaning |
| --- | --- |
| `doc_id` | Stable local document identifier. |
| `file_name` | PDF file name copied into `data/raw_pdfs/`. |
| `policy_name` | Policy name inferred from Chinese or romanized file name where possible; otherwise `TBD`. |
| `policy_family` | Broad policy family used later for routing and cross-policy retrieval. |
| `subdomain` | More specific workflow area, such as insider trading, whistleblowing, or derivatives. |
| `version` | Version found in the filename where possible; otherwise `TBD`. |
| `effective_date` | Date found in the filename where possible; otherwise `TBD`. |
| `confidentiality_label` | Source confidentiality label. Day 1 uses `Public` because the corpus is public PDFs. |
| `language` | Source language. |
| `source_type` | Source format and origin category. |

## Day 1 Data Handling

The `material/` folder is the original source location and remains untouched. `data/raw_pdfs/` contains working copies for ingestion in later days.
