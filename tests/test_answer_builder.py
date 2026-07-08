from pathlib import Path

from src.generation.answer_builder import (
    build_grounded_answer,
    hydrate_results,
    load_full_chunks,
)
from src.guardrails.disclaimers import DEMO_DISCLAIMER


def strong_result(**overrides):
    result = {
        "chunk_id": "TEST-001-article-001",
        "document_id": "WITS-001",
        "source_file": "policy.pdf",
        "file_name": "policy.pdf",
        "policy_name": "防範內線交易管理程序",
        "title": "防範內線交易管理程序",
        "chapter": "",
        "raw_article_marker": "第一條",
        "article": "第一條",
        "article_title": "目的",
        "fallback_type": "",
        "page_start": 1,
        "page_end": 1,
        "char_count": 100,
        "text": "公司重大資訊未公開前，相關人員應遵守保密義務，不得洩露或買賣有價證券。",
        "text_preview": "公司重大資訊未公開前，相關人員應遵守保密義務，不得洩露或買賣有價證券。",
        "final_score": 0.85,
        "keyword_boost": 0.10,
        "policy_boost": 0.20,
        "boilerplate_penalty": 0.0,
    }
    result.update(overrides)
    return result


def test_load_full_chunks_and_hydrate_use_full_text(tmp_path):
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(
        '{"chunk_id":"chunk-1","text":"完整政策文字比預覽更長","policy_name":"政策","article":"第一條"}\n',
        encoding="utf-8",
    )

    chunks = load_full_chunks(Path(chunks_path))
    hydrated = hydrate_results([{"chunk_id": "chunk-1", "text_preview": "短預覽"}], chunks)

    assert chunks["chunk-1"]["text"] == "完整政策文字比預覽更長"
    assert hydrated[0]["text"] == "完整政策文字比預覽更長"


def test_answer_includes_required_sections_disclaimer_and_citation_label():
    answer = build_grounded_answer("公司還沒公告重大資訊，可以買股票嗎？", [strong_result()])

    assert "## Direct Answer" in answer
    assert "## Policy-Based Explanation" in answer
    assert "## Relevant Evidence" in answer
    assert "## Citation Table" in answer
    assert "## Evidence Quality Note" in answer
    assert "## Demo Disclaimer" in answer
    assert DEMO_DISCLAIMER in answer
    assert "[C1]" in answer


def test_citation_table_maps_required_fields():
    answer = build_grounded_answer("公司還沒公告重大資訊，可以買股票嗎？", [strong_result()])

    assert "| [C1] | 防範內線交易管理程序 | 第一條 | 目的 | 1-1 | policy.pdf | TEST-001-article-001 |" in answer


def test_weak_evidence_returns_insufficient_response():
    weak = strong_result(
        policy_name="無關政策",
        article="第三條",
        article_title="一般事項",
        text="本辦法經董事會通過後實施，修訂時亦同。",
        final_score=0.10,
        keyword_boost=0.0,
        policy_boost=0.0,
        boilerplate_penalty=0.10,
    )

    answer = build_grounded_answer("這份文件有沒有說員工旅遊補助？", [weak])

    assert "The retrieved evidence is insufficient to answer this confidently." in answer
    assert "[C1]" in answer


def test_no_confident_answer_without_citations():
    no_citation = strong_result(source_file="", page_start="", page_end="")

    answer = build_grounded_answer("公司還沒公告重大資訊，可以買股票嗎？", [no_citation])

    assert "The retrieved evidence is insufficient to answer this confidently." in answer
    assert "this appears restricted" not in answer


def test_template_does_not_invent_policy_names_or_articles():
    result = strong_result(policy_name="舉報制度", article="第二條", article_title="受理範圍")

    answer = build_grounded_answer("我想檢舉內部舞弊，應該提供哪些資料？", [result])

    assert "舉報制度" in answer
    assert "第二條" in answer
    assert "不存在政策" not in answer
    assert "第九十九條" not in answer


def test_required_information_answer_uses_evidence_supported_items_only():
    result = strong_result(
        policy_name="舉報制度",
        article="第四條",
        article_title="檢舉程序",
        text="檢舉人應提供姓名、聯絡方式、被檢舉人姓名、檢舉內容及具體事證。不得要求提供本句以外的薪資資料。",
    )

    answer = build_grounded_answer("我想檢舉內部舞弊，應該提供哪些資料？", [result])

    assert "姓名" in answer
    assert "聯絡方式" in answer
    assert "具體事證" in answer
    assert "護照號碼" not in answer


def section_text(answer: str, heading: str) -> str:
    start = answer.index(f"## {heading}")
    next_heading = answer.find("\n\n## ", start + 1)
    if next_heading == -1:
        return answer[start:]
    return answer[start:next_heading]


def test_direct_answer_does_not_include_excessive_raw_evidence():
    long_text = "公司重大資訊未公開前，相關人員應遵守保密義務，不得洩露或買賣有價證券。" * 12
    answer = build_grounded_answer("公司還沒公告重大資訊，可以買股票嗎？", [strong_result(text=long_text)])
    direct = section_text(answer, "Direct Answer")

    assert len(direct) < 450
    assert long_text[:80] not in direct


def test_required_information_answer_uses_bullets():
    result = strong_result(
        policy_name="舉報制度",
        article="第四條",
        article_title="檢舉程序",
        text="檢舉人之姓名、身分證號碼及可聯絡到檢舉人之聯絡方式。被檢舉人之姓名或其他足茲識別被檢舉人身分特徵之資料。可供調查之具體事證。",
    )

    answer = build_grounded_answer("我想檢舉內部舞弊，應該提供哪些資料？", [result])
    direct = section_text(answer, "Direct Answer")

    assert "\n- 檢舉人之姓名" in direct
    assert "\n- 被檢舉人之姓名" in direct
    assert "\n- 可供調查之具體事證" in direct


def test_boilerplate_not_used_in_main_answer_when_substantive_exists():
    boilerplate = strong_result(
        chunk_id="TEST-boilerplate",
        article="第十九條",
        text="本辦法訂定於民國九十八年十一月五日經董事會核准公布後實施，修訂時亦同。",
        final_score=0.90,
        boilerplate_penalty=0.10,
    )
    substantive = strong_result(
        chunk_id="TEST-substantive",
        article="第十五條",
        policy_name="與特定公司集團企業及關係人交易管理辦法",
        text="本公司向關係人進行交易，達一定金額者，應將資料提交董事會通過後始得進行交易。",
        final_score=0.80,
        boilerplate_penalty=0.0,
    )

    answer = build_grounded_answer("關係人交易需要董事會核准嗎？", [boilerplate, substantive])
    main_body = section_text(answer, "Direct Answer") + section_text(answer, "Policy-Based Explanation")

    assert "第十五條" in main_body
    assert "第十九條" not in main_body
    assert "| [C1] |" in answer
    assert "| [C2] |" in answer


def test_derivative_restriction_wording_when_supported():
    result = strong_result(
        policy_name="從事衍生性商品交易處理規範",
        article="section-三",
        article_title="交易策略",
        text="公司從事衍生性商品交易應以避險策略為原則，不得從事非避險性交易。",
    )

    answer = build_grounded_answer("衍生性商品交易可以投機嗎？", [result])

    assert "non-hedging or speculative derivative trading appears restricted" in answer
