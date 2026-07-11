import json

from src.retrieval.chunker import article_blocks_for_document, chunk_document, detect_article_marker


def make_doc(text, document_id="TEST-001"):
    return {
        "document_id": document_id,
        "source_file": "/tmp/test.pdf",
        "file_name": "test.pdf",
        "policy_name": "測試政策",
        "title": "測試政策",
        "extraction_status": "success",
        "pages": [{"page_number": 1, "text": text, "char_count": len(text)}],
    }


def test_detects_basic_article_marker():
    marker = detect_article_marker("第一條 目的")
    assert marker["raw_article_marker"] == "第一條"
    assert marker["article"] == "第一條"


def test_detects_sub_article_marker():
    marker = detect_article_marker("第十條之一：處理程序")
    assert marker["article"] == "第十條之一"


def test_detects_spaced_article_marker():
    marker = detect_article_marker("第 一 條：目的")
    assert marker["raw_article_marker"] == "第 一 條："
    assert marker["article"] == "第一條"


def test_detects_spaced_sub_article_marker():
    marker = detect_article_marker("第 十 二 條 之 一：補充規定")
    assert marker["raw_article_marker"] == "第 十 二 條 之 一："
    assert marker["article"] == "第十二條之一"


def test_does_not_detect_article_reference_inside_sentence():
    assert detect_article_marker("依第十二條規定辦理") is None


def test_article_title_on_same_line():
    blocks = article_blocks_for_document(make_doc("第一條 目的\n本文內容"))
    assert blocks[0].article_title == "目的"


def test_article_title_on_next_non_empty_line():
    blocks = article_blocks_for_document(make_doc("第一條\n\n目的及法源依據\n本文內容"))
    assert blocks[0].article_title == "目的及法源依據"


def test_chunks_preserve_citation_metadata():
    chunks, used_fallback = chunk_document(make_doc("第一條 目的\n本文內容", document_id="WITS-X"))
    assert not used_fallback
    chunk = chunks[0]
    assert chunk["document_id"] == "WITS-X"
    assert chunk["policy_name"] == "測試政策"
    assert chunk["page_start"] == 1
    assert chunk["page_end"] == 1
    assert chunk["raw_article_marker"] == "第一條"
    assert chunk["article"] == "第一條"


def test_chapter_is_context_not_chunk():
    chunks, used_fallback = chunk_document(make_doc("第一章 總則\n第一條 目的\n本文內容"))
    assert not used_fallback
    assert len(chunks) == 1
    assert chunks[0]["chapter"] == "第一章 總則"
    assert chunks[0]["article"] == "第一條"


def test_split_chunks_preserve_article_metadata():
    long_text = "第一條 目的\n" + "長內容" * 900
    chunks, _ = chunk_document(make_doc(long_text), max_chars=1000)
    assert len(chunks) > 1
    assert all(chunk["article"] == "第一條" for chunk in chunks)
    assert all(chunk["is_split_part"] for chunk in chunks)
    assert chunks[1]["part_index"] == 2
    assert chunks[1]["parent_article_id"] == "TEST-001-article-001"


def test_section_fallback_chunks_when_no_articles():
    doc = make_doc("一、目的\n目的內容\n二、適用範圍\n範圍內容")
    chunks, fallback_type = chunk_document(doc)
    assert fallback_type == "section"
    assert len(chunks) == 2
    assert chunks[0]["chunk_id"] == "TEST-001-section-001"
    assert chunks[0]["raw_article_marker"] == "一、"
    assert chunks[0]["article"] == "section-一"
    assert chunks[0]["article_title"] == "目的"
    assert chunks[0]["fallback_type"] == "section"
    assert chunks[0]["document_id"] == "TEST-001"
    assert chunks[0]["policy_name"] == "測試政策"
    assert chunks[0]["page_start"] == 1
    assert chunks[0]["page_end"] == 1
    assert chunks[1]["raw_article_marker"] == "二、"
    assert chunks[1]["article_title"] == "適用範圍"


def test_section_reference_inside_sentence_is_not_marker():
    doc = make_doc("本規範依一、目的所述辦理，非標題行")
    chunks, fallback_type = chunk_document(doc)
    assert fallback_type == "page"
    assert chunks[0]["fallback_type"] == "page"
