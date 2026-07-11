"""Fixed lightweight evaluation examples for the Streamlit demo."""

from __future__ import annotations

from typing import Any


EVALUATION_EXAMPLES = [
    {
        "query": "公司還沒公告重大資訊，可以買股票嗎？",
        "expected_risks": ["High"],
        "expected_categories": ["Insider Trading / Material Non-Public Information"],
    },
    {
        "query": "我想檢舉內部舞弊，應該提供哪些資料？",
        "expected_risks": ["High", "Medium"],
        "expected_categories": ["Whistleblowing / Fraud Reporting"],
    },
    {
        "query": "衍生性商品交易可以投機嗎？",
        "expected_risks": ["High"],
        "expected_categories": ["Derivative Trading / Hedging Control"],
    },
    {
        "query": "關係人交易需要董事會核准嗎？",
        "expected_risks": ["Medium", "High"],
        "expected_categories": ["Related-Party Transaction", "Board Approval / Governance Procedure"],
    },
    {
        "query": "這份文件有沒有說員工旅遊補助？",
        "expected_risks": ["Insufficient Evidence"],
        "expected_categories": ["Unknown or Insufficient Evidence"],
    },
]


def evaluate_response(example: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    actual_risk = response.get("risk_level", "") if isinstance(response, dict) else ""
    actual_category = response.get("risk_category", "") if isinstance(response, dict) else ""
    expected_risks = example.get("expected_risks", [])
    expected_categories = example.get("expected_categories", [])
    risk_pass = actual_risk in expected_risks
    category_pass = actual_category in expected_categories
    return {
        "query": example.get("query", ""),
        "expected_risk": " or ".join(expected_risks),
        "actual_risk": actual_risk,
        "expected_category": " or ".join(expected_categories),
        "actual_category": actual_category,
        "pass": risk_pass and category_pass,
        "evidence_quality": response.get("evidence_quality", "") if isinstance(response, dict) else "",
    }


def summarize_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    passed = sum(1 for row in rows if row.get("pass"))
    pass_rate = passed / total if total else 0.0
    return {"total_examples": total, "passed_examples": passed, "pass_rate": pass_rate}
