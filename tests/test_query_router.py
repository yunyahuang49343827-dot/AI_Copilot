from src.retrieval.query_router import (
    CATEGORY_ASSET,
    CATEGORY_BOARD_GOVERNANCE,
    CATEGORY_DERIVATIVES,
    CATEGORY_INSIDER_TRADING,
    CATEGORY_LENDING_ENDORSEMENT_GUARANTEE,
    CATEGORY_RELATED_PARTY,
    CATEGORY_UNKNOWN,
    CATEGORY_WHISTLEBLOWING,
    classify_query_route,
)


def test_insider_trading_query_routes_to_insider_document():
    route = classify_query_route("員工知道未公開資訊時，可以進行股票交易嗎？")

    assert route.primary_category == CATEGORY_INSIDER_TRADING
    assert "WITS-001" in route.candidate_documents
    assert route.confidence > 0


def test_asset_acquisition_disposal_query_routes_to_wits004():
    route = classify_query_route("公司取得或處分資產時，需要哪些程序？")

    assert route.primary_category == CATEGORY_ASSET
    assert route.candidate_documents == ["WITS-004"]


def test_related_party_query_routes_to_wits005():
    route = classify_query_route("董事控制的公司是否屬於關係人交易？")

    assert route.primary_category == CATEGORY_RELATED_PARTY
    assert "WITS-005" in route.candidate_documents


def test_board_governance_query_routes_to_wits010_and_article_12():
    route = classify_query_route("重大財務業務需要董事會核准或決議嗎？")

    assert route.primary_category == CATEGORY_BOARD_GOVERNANCE
    assert "WITS-010" in route.candidate_documents
    assert "第十二條" in route.candidate_articles


def test_derivatives_query_routes_to_wits006():
    route = classify_query_route("衍生性商品交易是否可以用於避險？")

    assert route.primary_category == CATEGORY_DERIVATIVES
    assert route.candidate_documents == ["WITS-006"]


def test_funds_lending_query_routes_to_lending_endorsement_guarantee_and_wits008():
    route = classify_query_route("公司資金貸與他人時需要哪些程序？")

    assert route.primary_category == CATEGORY_LENDING_ENDORSEMENT_GUARANTEE
    assert "WITS-008" in route.candidate_documents


def test_endorsement_guarantee_query_routes_to_lending_endorsement_guarantee_and_wits007():
    route = classify_query_route("替子公司背書保證或對外保證需要注意什麼？")

    assert route.primary_category == CATEGORY_LENDING_ENDORSEMENT_GUARANTEE
    assert "WITS-007" in route.candidate_documents


def test_whistleblowing_query_routes_to_whistleblowing():
    route = classify_query_route("我想檢舉內部舞弊或不法行為，應該如何通報？")

    assert route.primary_category == CATEGORY_WHISTLEBLOWING
    assert "WITS-003" in route.candidate_documents


def test_mixed_asset_related_party_board_query_includes_expected_documents():
    route = classify_query_route("和董事控制的公司進行資產交易，應該同時看哪些政策？")

    assert route.is_mixed_policy is True
    assert route.primary_category == CATEGORY_ASSET
    assert route.candidate_categories == [CATEGORY_RELATED_PARTY, CATEGORY_ASSET, CATEGORY_BOARD_GOVERNANCE]
    assert route.candidate_documents == ["WITS-005", "WITS-004", "WITS-010"]
    assert "董事控制的公司" in route.matched_terms
    assert "資產交易" in route.matched_terms


def test_mixed_asset_related_party_cross_check_query_includes_expected_documents():
    route = classify_query_route("資產交易如果同時涉及關係人，應該如何交叉檢查？")

    assert route.is_mixed_policy is True
    assert route.primary_category == CATEGORY_ASSET
    assert "WITS-004" in route.candidate_documents
    assert "WITS-005" in route.candidate_documents
    assert "WITS-010" in route.candidate_documents
    assert route.confidence == 0.95


def test_generic_unsupported_query_returns_unknown_low_confidence_route():
    route = classify_query_route("員工旅遊補助如何申請？")

    assert route.primary_category == CATEGORY_UNKNOWN
    assert route.candidate_categories == []
    assert route.candidate_documents == []
    assert route.is_mixed_policy is False
    assert route.confidence == 0.0
