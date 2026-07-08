from src.guardrails.evidence_checks import (
    assess_evidence,
    boilerplate_ratio,
    expanded_query_terms,
    has_complete_citation,
    query_overlap,
)


def result(**overrides):
    base = {
        "chunk_id": "chunk-1",
        "policy_name": "內部重大資訊處理作業程序",
        "article": "第五條",
        "article_title": "重大資訊處理",
        "page_start": 1,
        "page_end": 2,
        "source_file": "policy.pdf",
        "text": "重大資訊及未公開資訊應依程序處理。",
        "text_preview": "重大資訊及未公開資訊應依程序處理。",
        "final_score": 0.80,
        "keyword_boost": 0.10,
        "policy_boost": 0.20,
        "boilerplate_penalty": 0.0,
    }
    base.update(overrides)
    return base


def test_has_complete_citation_requires_core_fields():
    assert has_complete_citation(result())
    assert not has_complete_citation(result(source_file=""))


def test_expanded_query_terms_adds_synonyms():
    terms = expanded_query_terms("我想檢舉內部舞弊")

    assert "舉報" in terms
    assert "通報" in terms
    assert "不法" in terms


def test_query_overlap_uses_metadata_and_synonyms():
    overlap = query_overlap(
        "公司還沒公告重大資訊，可以買股票嗎？",
        result(policy_name="防範內線交易管理程序", text="未公開資訊及有價證券買賣應依規定辦理。"),
    )

    assert overlap > 0


def test_boilerplate_only_evidence_is_weak():
    boilerplate = [
        result(
            text="本程序經董事會通過後實施，修訂時亦同。",
            final_score=0.85,
            keyword_boost=0.10,
            policy_boost=0.20,
            boilerplate_penalty=0.10,
        ),
        result(
            chunk_id="chunk-2",
            text="第一次修訂，第二次修訂。",
            boilerplate_penalty=0.10,
        ),
    ]

    assessment = assess_evidence("內線交易", boilerplate)

    assert not assessment.is_sufficient
    assert "boilerplate" in " ".join(assessment.reasons)


def test_substantive_boosted_evidence_is_sufficient():
    assessment = assess_evidence(
        "公司還沒公告重大資訊，可以買股票嗎？",
        [result(policy_name="防範內線交易管理程序", text="重大資訊未公開前，不得洩露或買賣有價證券。")],
    )

    assert assessment.is_sufficient


def test_low_score_without_boost_is_weak():
    assessment = assess_evidence(
        "員工旅遊補助",
        [result(policy_name="無關政策", text="董事會議事程序。", final_score=0.05, keyword_boost=0.0, policy_boost=0.0)],
    )

    assert not assessment.is_sufficient


def test_asset_policy_evidence_is_sufficient_without_boost_for_asset_procedure_query():
    assessment = assess_evidence(
        "取得或處分重大資產需要遵循什麼程序？",
        [
            result(
                document_id="WITS-004",
                policy_name="取得或處分資產處理程序",
                article_title="取得或處分資產評估及作業程序",
                text="取得或處分資產應依評估及作業程序辦理。",
                text_preview="取得或處分資產應依評估及作業程序辦理。",
                keyword_boost=0.0,
                policy_boost=0.0,
            )
        ],
    )

    assert assessment.is_sufficient


def test_asset_policy_evidence_is_sufficient_without_boost_for_real_estate_valuation_query():
    assessment = assess_evidence(
        "出售不動產或設備時，如何確認價格合理性？",
        [
            result(
                document_id="WITS-004",
                policy_name="取得或處分資產處理程序",
                article_title="專業估價機構之估價報告",
                text="取得或處分不動產或設備時，應評估交易價格合理性並取得估價資料。",
                text_preview="取得或處分不動產或設備時，應評估交易價格合理性並取得估價資料。",
                keyword_boost=0.0,
                policy_boost=0.0,
            )
        ],
    )

    assert assessment.is_sufficient


def test_asset_policy_evidence_does_not_make_unrelated_query_sufficient():
    assessment = assess_evidence(
        "公司政策有沒有提到員工旅遊補助？",
        [
            result(
                document_id="WITS-004",
                policy_name="取得或處分資產處理程序",
                article_title="取得或處分資產評估及作業程序",
                text="取得或處分資產應依評估及作業程序辦理。",
                text_preview="取得或處分資產應依評估及作業程序辦理。",
                keyword_boost=0.0,
                policy_boost=0.0,
            )
        ],
    )

    assert not assessment.is_sufficient
