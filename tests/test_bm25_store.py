from src.retrieval.bm25_store import build_bm25_payload, search_payload, tokenize


def test_tokenizer_preserves_compliance_terms():
    tokens = tokenize("公司涉及內線交易與重大資訊，應可檢舉舞弊")
    assert "內線交易" in tokens
    assert "重大資訊" in tokens
    assert "檢舉" in tokens
    assert "舞弊" in tokens


def test_tokenizer_creates_chinese_bigrams():
    tokens = tokenize("風險管理")
    assert "風險管理" in tokens
    assert "風險" in tokens
    assert "管理" in tokens


def test_tokenizer_preserves_english_terms():
    tokens = tokenize("Forward Option Swap Future API")
    assert "forward" in tokens
    assert "option" in tokens
    assert "swap" in tokens
    assert "future" in tokens
    assert "api" in tokens


def test_bm25_payload_returns_keyword_relevant_results():
    chunks = [
        {"chunk_id": "a", "policy_name": "防範內線交易管理程序", "article": "第一條", "article_title": "目的", "text": "禁止內線交易", "page_start": 1, "page_end": 1},
        {"chunk_id": "b", "policy_name": "風險管理政策與程序", "article": "第一條", "article_title": "目的", "text": "風險管理報告", "page_start": 1, "page_end": 1},
    ]
    payload = build_bm25_payload(chunks)
    results = search_payload(payload, "內線交易", top_k=1)
    assert results[0]["chunk_id"] == "a"
    assert results[0]["bm25_score"] >= 0
