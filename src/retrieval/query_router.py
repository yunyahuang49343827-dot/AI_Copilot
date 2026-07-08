"""Isolated deterministic query router for future retrieval experiments."""

from __future__ import annotations

from dataclasses import dataclass


CATEGORY_INSIDER_TRADING = "insider_trading"
CATEGORY_WHISTLEBLOWING = "whistleblowing"
CATEGORY_RELATED_PARTY = "related_party_transaction"
CATEGORY_ASSET = "asset_acquisition_disposal"
CATEGORY_DERIVATIVES = "derivatives"
CATEGORY_LENDING_ENDORSEMENT_GUARANTEE = "lending_endorsement_guarantee"
CATEGORY_BOARD_GOVERNANCE = "board_governance"
CATEGORY_UNKNOWN = "unknown_or_insufficient_evidence"

PRIMARY_CATEGORY_PRECEDENCE = [
    CATEGORY_ASSET,
    CATEGORY_RELATED_PARTY,
    CATEGORY_DERIVATIVES,
    CATEGORY_LENDING_ENDORSEMENT_GUARANTEE,
    CATEGORY_INSIDER_TRADING,
    CATEGORY_WHISTLEBLOWING,
    CATEGORY_BOARD_GOVERNANCE,
]

CATEGORY_RULES = [
    {
        "category": CATEGORY_INSIDER_TRADING,
        "documents": ["WITS-001"],
        "articles": [],
        "keywords": [
            "insider trading",
            "material information",
            "unpublished information",
            "內線交易",
            "重大消息",
            "未公開資訊",
            "有價證券交易",
            "股票交易",
        ],
    },
    {
        "category": CATEGORY_WHISTLEBLOWING,
        "documents": ["WITS-003"],
        "articles": [],
        "keywords": [
            "whistleblowing",
            "report fraud",
            "fraud reporting",
            "舉報",
            "檢舉",
            "舞弊",
            "不法",
            "違法",
            "通報",
        ],
    },
    {
        "category": CATEGORY_RELATED_PARTY,
        "documents": ["WITS-005"],
        "articles": [],
        "keywords": [
            "related party",
            "related-party",
            "關係人",
            "關係企業",
            "董事控制的公司",
            "董事控制",
            "利害關係人",
        ],
    },
    {
        "category": CATEGORY_ASSET,
        "documents": ["WITS-004"],
        "articles": [],
        "keywords": [
            "asset acquisition",
            "asset disposal",
            "acquisition or disposal of assets",
            "資產取得",
            "資產處分",
            "資產交易",
            "取得或處分資產",
            "處分資產",
            "取得資產",
        ],
    },
    {
        "category": CATEGORY_DERIVATIVES,
        "documents": ["WITS-006"],
        "articles": [],
        "keywords": [
            "derivatives",
            "hedging",
            "derivative trading",
            "衍生性商品",
            "避險",
            "期貨",
            "選擇權",
        ],
    },
    {
        "category": CATEGORY_LENDING_ENDORSEMENT_GUARANTEE,
        "documents": ["WITS-007", "WITS-008"],
        "articles": [],
        "keywords": [
            "funds lending",
            "endorsement",
            "guarantee",
            "loaning funds",
            "資金貸與",
            "背書",
            "保證",
            "背書保證",
            "對外保證",
        ],
    },
    {
        "category": CATEGORY_BOARD_GOVERNANCE,
        "documents": ["WITS-010"],
        "articles": ["第十二條"],
        "keywords": [
            "board",
            "board approval",
            "board meeting",
            "governance",
            "董事會",
            "董事",
            "核准",
            "決議",
            "重大財務業務",
            "交叉檢查",
            "同時看哪些政策",
            "內部重大資訊",
            "重大程序",
        ],
    },
]


@dataclass(frozen=True)
class QueryRoute:
    primary_category: str | None
    candidate_categories: list[str]
    candidate_documents: list[str]
    candidate_articles: list[str]
    is_mixed_policy: bool
    confidence: float
    matched_terms: list[str]


def _dedupe_stable(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _query_contains(query: str, keyword: str) -> bool:
    return keyword.lower() in query.lower()


def _remove_subphrase_matches(matches: list[str]) -> list[str]:
    kept: list[str] = []
    for keyword in matches:
        if any(keyword != other and keyword.lower() in other.lower() for other in matches):
            continue
        kept.append(keyword)
    return kept


def _confidence(match_counts: dict[str, int], is_mixed_policy: bool) -> float:
    if not match_counts:
        return 0.0
    if is_mixed_policy and len(match_counts) >= 3:
        return 0.95
    if is_mixed_policy:
        return 0.85
    count = next(iter(match_counts.values()))
    return 0.7 if count >= 2 else 0.5


def _primary_category(match_counts: dict[str, int]) -> str:
    if not match_counts:
        # Keep unknown explicit so downstream experiments can distinguish "no route" from a missing field.
        return CATEGORY_UNKNOWN
    return sorted(
        match_counts,
        key=lambda category: (-match_counts[category], PRIMARY_CATEGORY_PRECEDENCE.index(category)),
    )[0]


def classify_query_route(query: str) -> QueryRoute:
    normalized_query = query or ""
    match_counts: dict[str, int] = {}
    candidate_categories: list[str] = []
    candidate_documents: list[str] = []
    candidate_articles: list[str] = []
    matched_terms: list[str] = []
    raw_matches_by_category: dict[str, list[str]] = {}

    for rule in CATEGORY_RULES:
        matches = [keyword for keyword in rule["keywords"] if _query_contains(normalized_query, keyword)]
        if not matches:
            continue
        raw_matches_by_category[str(rule["category"])] = matches

    retained_terms = set(_remove_subphrase_matches([term for matches in raw_matches_by_category.values() for term in matches]))

    for rule in CATEGORY_RULES:
        category = str(rule["category"])
        matches = [keyword for keyword in raw_matches_by_category.get(category, []) if keyword in retained_terms]
        if not matches:
            continue
        match_counts[category] = len(matches)
        candidate_categories.append(category)
        candidate_documents.extend(rule["documents"])
        candidate_articles.extend(rule["articles"])
        matched_terms.extend(matches)

    candidate_categories = _dedupe_stable(candidate_categories)
    candidate_documents = _dedupe_stable(candidate_documents)
    candidate_articles = _dedupe_stable(candidate_articles)
    matched_terms = _dedupe_stable(matched_terms)
    is_mixed_policy = len(candidate_categories) >= 2

    return QueryRoute(
        primary_category=_primary_category(match_counts),
        candidate_categories=candidate_categories,
        candidate_documents=candidate_documents,
        candidate_articles=candidate_articles,
        is_mixed_policy=is_mixed_policy,
        confidence=_confidence(match_counts, is_mixed_policy),
        matched_terms=matched_terms,
    )
