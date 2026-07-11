from src.retrieval.query_router import CATEGORY_ASSET, CATEGORY_UNKNOWN, QueryRoute
from src.retrieval.reranker import rerank_results


def route(**overrides):
    base = {
        "primary_category": CATEGORY_ASSET,
        "candidate_categories": [CATEGORY_ASSET],
        "candidate_documents": ["WITS-004"],
        "candidate_articles": ["第十二條"],
        "is_mixed_policy": False,
        "confidence": 0.7,
        "matched_terms": ["資產交易"],
    }
    base.update(overrides)
    return QueryRoute(**base)


def result(document_id="WITS-999", final_score=0.5, **overrides):
    base = {
        "chunk_id": f"{document_id}-chunk",
        "document_id": document_id,
        "article": "第一條",
        "policy_name": "其他政策",
        "article_title": "一般事項",
        "text_preview": "一般治理內容",
        "final_score": final_score,
    }
    base.update(overrides)
    return base


def test_reranker_preserves_all_input_results():
    results = [result("WITS-999"), result("WITS-004")]

    reranked = rerank_results("資產交易", results, route())

    assert len(reranked) == len(results)
    assert {item["chunk_id"] for item in reranked} == {item["chunk_id"] for item in results}


def test_reranker_does_not_mutate_original_input_dicts():
    results = [result("WITS-004", policy_name="取得或處分資產處理程序")]
    original = dict(results[0])

    rerank_results("資產交易", results, route())

    assert results[0] == original
    assert "rerank_score" not in results[0]
    assert "original_rank" not in results[0]


def test_matching_candidate_document_receives_higher_rerank_score():
    reranked = rerank_results(
        "資產交易",
        [
            result("WITS-004", final_score=0.4),
            result("WITS-999", final_score=0.4),
        ],
        route(),
    )

    scores = {item["document_id"]: item["rerank_score"] for item in reranked}
    assert scores["WITS-004"] > scores["WITS-999"]
    assert "matched_candidate_document" in reranked[0]["rerank_reason"]


def test_candidate_docs_can_move_above_unrelated_docs_when_scores_are_close():
    reranked = rerank_results(
        "資產交易",
        [
            result("WITS-999", final_score=0.6),
            result("WITS-004", final_score=0.5, policy_name="取得或處分資產處理程序"),
        ],
        route(),
    )

    assert reranked[0]["document_id"] == "WITS-004"
    assert reranked[0]["rerank_score"] > reranked[1]["rerank_score"]


def test_deterministic_ordering_for_ties_uses_original_rank():
    tie_route = route(
        primary_category=CATEGORY_UNKNOWN,
        candidate_categories=[],
        candidate_documents=[],
        candidate_articles=[],
        matched_terms=[],
        confidence=0.0,
    )
    results = [result("WITS-A", final_score=0.5), result("WITS-B", final_score=0.5)]

    reranked = rerank_results("unsupported query", results, tie_route)

    assert [item["document_id"] for item in reranked] == ["WITS-A", "WITS-B"]
    assert [item["original_rank"] for item in reranked] == [1, 2]


def test_route_none_calls_router_and_returns_valid_reranked_results():
    reranked = rerank_results(
        "資產交易需要檢查哪些程序？",
        [result("WITS-004", final_score=0.2), result("WITS-999", final_score=0.25)],
        route=None,
    )

    assert reranked[0]["document_id"] == "WITS-004"
    assert all("rerank_score" in item for item in reranked)


def test_missing_final_score_is_handled_safely():
    missing_score = result("WITS-004")
    del missing_score["final_score"]

    reranked = rerank_results("資產交易", [missing_score], route())

    assert reranked[0]["rerank_score"] >= 0
    assert reranked[0].get("final_score") is None


def test_rerank_output_includes_score_reason_and_original_rank():
    reranked = rerank_results("資產交易", [result("WITS-004")], route())

    assert isinstance(reranked[0]["rerank_score"], float)
    assert isinstance(reranked[0]["rerank_reason"], str)
    assert reranked[0]["original_rank"] == 1
