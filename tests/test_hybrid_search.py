import src.retrieval.hybrid_search as hybrid_module
from src.retrieval.hybrid_search import compute_boilerplate_penalty, compute_boosts, format_hybrid_result, merge_results, min_max_normalize


def base_result(chunk_id="a", policy="防範內線交易管理程序", title="目的"):
    return {
        "chunk_id": chunk_id,
        "document_id": "WITS-X",
        "policy_name": policy,
        "article": "第一條",
        "article_title": title,
        "page_start": 1,
        "page_end": 1,
        "source_file": "source.pdf",
        "fallback_type": "",
        "text": "重大資訊 未公開 買股票",
        "vector_distance": 0.2,
    }


def test_min_max_normalize_stable_for_equal_scores():
    assert min_max_normalize([2.0, 2.0]) == [1.0, 1.0]
    assert min_max_normalize([0.0, 0.0]) == [0.0, 0.0]


def test_hybrid_merge_deduplicates_by_chunk_id():
    vector = [base_result("same")]
    bm25 = [{**base_result("same"), "bm25_score": 3.0}]
    merged = merge_results(vector, bm25, "重大資訊")
    assert len(merged) == 1
    assert merged[0]["chunk_id"] == "same"
    assert merged[0]["bm25_score"] == 3.0


def test_keyword_boosts_apply_for_major_information():
    keyword_boost, policy_boost = compute_boosts("重大資訊 買股票", base_result())
    assert keyword_boost > 0
    assert policy_boost > 0


def test_keyword_boosts_apply_for_whistleblowing():
    result = base_result(policy="舉報制度")
    result["text"] = "檢舉 舞弊 通報"
    keyword_boost, policy_boost = compute_boosts("我要檢舉內部舞弊", result)
    assert keyword_boost > 0
    assert policy_boost > 0


def test_derivative_section_boost_applies_to_trading_strategy():
    result = base_result(policy="從事衍生性商品交易處理規範", title="交易策略")
    result["text"] = "非避險性交易 投機 衍生性商品"
    keyword_boost, policy_boost = compute_boosts("衍生性商品交易可以投機嗎", result)
    assert keyword_boost > 0
    assert policy_boost >= 0.20


def test_formatting_includes_citation_and_scores():
    result = {
        **base_result(),
        "final_score": 0.9,
        "vector_score_normalized": 0.8,
        "bm25_score": 2.0,
        "bm25_score_normalized": 0.7,
        "keyword_boost": 0.08,
        "policy_boost": 0.14,
        "boilerplate_penalty": 0.12,
        "text_preview": "preview",
    }
    formatted = format_hybrid_result(result, 1)
    assert "Final Score" in formatted
    assert "Chunk ID: a" in formatted
    assert "Policy: 防範內線交易管理程序" in formatted
    assert "Boilerplate Penalty: 0.1200" in formatted
    assert "Pages: 1-1" in formatted


def test_boilerplate_chunks_receive_penalty():
    result = base_result(title="其他事項")
    result["text"] = "本程序訂定於民國九十八年十一月五日，第一次修訂於民國一百年。"
    assert compute_boilerplate_penalty(result) > 0


def test_substantive_chunks_do_not_receive_boilerplate_penalty():
    result = base_result(title="本程序第二條第二款所指之重大消息係指")
    result["text"] = "重大消息明確後，未公開前或公開後十八小時內，不得買賣股票。"
    assert compute_boilerplate_penalty(result) == 0.0


def test_merge_subtracts_boilerplate_penalty():
    substantive = base_result("substantive")
    boilerplate = base_result("boilerplate", title="其他事項")
    boilerplate["text"] = "本程序訂定於民國九十八年，第一次修訂於民國一百年。"
    merged = merge_results([substantive, boilerplate], [], "重大資訊")
    penalties = {item["chunk_id"]: item["boilerplate_penalty"] for item in merged}
    assert penalties["substantive"] == 0.0
    assert penalties["boilerplate"] > 0


def vector_payload(records):
    return {
        "documents": [[record.get("text", "") for record in records]],
        "metadatas": [[{key: value for key, value in record.items() if key != "text"} for record in records]],
        "distances": [[record.get("vector_distance", 0.2) for record in records]],
    }


def test_hybrid_search_default_does_not_call_reranker(monkeypatch):
    records = [
        base_result("first", policy="取得或處分資產處理程序"),
        base_result("second", policy="其他政策"),
    ]
    records[0]["vector_distance"] = 0.1
    records[1]["vector_distance"] = 0.5

    monkeypatch.setattr(hybrid_module, "ensure_chroma_available", lambda: None)
    monkeypatch.setattr(hybrid_module, "vector_query_index", lambda query, top_k: vector_payload(records))
    monkeypatch.setattr(hybrid_module, "bm25_query_index", lambda query, top_k: [])

    def fail_if_called(query, results):
        raise AssertionError("reranker should not be called when enable_reranker is False")

    monkeypatch.setattr(hybrid_module, "rerank_results", fail_if_called)

    results = hybrid_module.hybrid_search("資產交易", top_k=1)

    assert len(results) == 1
    assert results[0]["chunk_id"] == "first"
    assert "rerank_score" not in results[0]


def test_hybrid_search_calls_reranker_when_feature_flag_enabled(monkeypatch):
    records = [
        base_result("unrelated", policy="其他政策"),
        base_result("asset", policy="取得或處分資產處理程序"),
    ]
    records[0]["vector_distance"] = 0.1
    records[1]["vector_distance"] = 0.5
    calls = []

    monkeypatch.setattr(hybrid_module, "ensure_chroma_available", lambda: None)
    monkeypatch.setattr(hybrid_module, "vector_query_index", lambda query, top_k: vector_payload(records))
    monkeypatch.setattr(hybrid_module, "bm25_query_index", lambda query, top_k: [])

    def fake_rerank(query, results):
        calls.append({"query": query, "candidate_count": len(results)})
        reranked = [dict(result) for result in results]
        for index, result in enumerate(reranked, start=1):
            result["original_rank"] = index
            result["rerank_score"] = 1.0 if result["chunk_id"] == "asset" else 0.5
            result["rerank_reason"] = "test"
        return sorted(reranked, key=lambda result: result["rerank_score"], reverse=True)

    monkeypatch.setattr(hybrid_module, "rerank_results", fake_rerank)

    results = hybrid_module.hybrid_search("資產交易", top_k=1, enable_reranker=True, rerank_candidate_k=2)

    assert calls == [{"query": "資產交易", "candidate_count": 2}]
    assert results[0]["chunk_id"] == "asset"
    assert results[0]["rerank_reason"] == "test"


def test_hybrid_search_rerank_candidate_k_expands_retrieval_only_when_enabled(monkeypatch):
    requested_top_ks = []

    monkeypatch.setattr(hybrid_module, "ensure_chroma_available", lambda: None)

    def fake_vector_query(query, top_k):
        requested_top_ks.append(top_k)
        return vector_payload([base_result("candidate")])

    monkeypatch.setattr(hybrid_module, "vector_query_index", fake_vector_query)
    monkeypatch.setattr(hybrid_module, "bm25_query_index", lambda query, top_k: [])
    monkeypatch.setattr(hybrid_module, "rerank_results", lambda query, results: results)

    hybrid_module.hybrid_search("資產交易", top_k=3, enable_reranker=False, rerank_candidate_k=10)
    hybrid_module.hybrid_search("資產交易", top_k=3, enable_reranker=True, rerank_candidate_k=10)

    assert requested_top_ks == [20, 40]
