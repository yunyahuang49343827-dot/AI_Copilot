from src.agent.workflow_advisor import advise_workflow


def result(**overrides):
    base = {
        "chunk_id": "TEST-001",
        "document_id": "WITS-001",
        "source_file": "policy.pdf",
        "file_name": "policy.pdf",
        "policy_name": "防範內線交易管理程序",
        "title": "防範內線交易管理程序",
        "chapter": "",
        "raw_article_marker": "第五條",
        "article": "第五條",
        "article_title": "重大消息",
        "fallback_type": "",
        "page_start": 1,
        "page_end": 2,
        "char_count": 100,
        "text": "重大資訊未公開前，不得洩露或買賣有價證券。",
        "text_preview": "重大資訊未公開前，不得洩露或買賣有價證券。",
        "final_score": 0.90,
        "keyword_boost": 0.10,
        "policy_boost": 0.20,
        "boilerplate_penalty": 0.0,
    }
    base.update(overrides)
    return base


def retriever(results):
    return lambda query, top_k: results


def test_workflow_output_includes_sections_citation_table_and_disclaimer():
    output = advise_workflow("公司還沒公告重大資訊，可以買股票嗎？", retriever=retriever([result()]))

    assert "## Grounded Answer" in output
    assert "## Risk Triage" in output
    assert "## Workflow Checklist" in output
    assert "## Citation Table" in output
    assert "## Evidence Quality Note" in output
    assert "## Demo Disclaimer" in output
    assert "Risk level is a demo triage priority" in output
    assert "| [C1] | 防範內線交易管理程序" in output


def test_workflow_reasoning_includes_citation_labels_when_sufficient():
    output = advise_workflow("公司還沒公告重大資訊，可以買股票嗎？", retriever=retriever([result()]))

    assert "- Risk Level: High" in output
    assert "- Risk Category: Insider Trading / Material Non-Public Information" in output
    assert "- Reasoning:" in output and "[C1]" in output


def test_unsupported_query_returns_insufficient_without_invented_checklist():
    weak = result(policy_name="無關政策", article="第六條", article_title="職權", text="員工溝通管道。", final_score=0.05, keyword_boost=0.0, policy_boost=0.0)

    output = advise_workflow("這份文件有沒有說員工旅遊補助？", retriever=retriever([weak]))

    assert "- Risk Level: Insufficient Evidence" in output
    assert "- Risk Category: Unknown or Insufficient Evidence" in output
    assert "Do not rely on an automated workflow checklist" in output
    assert "board approval check" not in output.lower()


def test_asset_workflow_with_wits004_evidence_is_not_insufficient():
    asset = result(
        chunk_id="TEST-ASSET-001",
        document_id="WITS-004",
        policy_name="取得或處分資產處理程序",
        article="第四條",
        article_title="取得或處分資產評估及作業程序",
        text="取得或處分重大資產時，應依評估及作業程序辦理。",
        text_preview="取得或處分重大資產時，應依評估及作業程序辦理。",
        final_score=0.75,
        keyword_boost=0.0,
        policy_boost=0.0,
    )

    output = advise_workflow("公司取得或處分重大資產時，需要走哪些程序？", retriever=retriever([asset]))

    assert "- Risk Level: Insufficient Evidence" not in output
    assert "- Risk Category: Asset Acquisition or Disposal" in output


def test_asset_workflow_with_wits004_evidence_generates_asset_checklist():
    asset = result(
        chunk_id="TEST-ASSET-002",
        document_id="WITS-004",
        policy_name="取得或處分資產處理程序",
        article="第十條",
        article_title="專業估價機構之估價報告",
        text="出售不動產或設備時，應評估交易價格合理性並取得估價資料。",
        text_preview="出售不動產或設備時，應評估交易價格合理性並取得估價資料。",
        final_score=0.75,
        keyword_boost=0.0,
        policy_boost=0.0,
    )

    output = advise_workflow("出售不動產或設備時，如何確認價格合理性？", retriever=retriever([asset]))

    assert "- Risk Category: Asset Acquisition or Disposal" in output
    assert "Identify the asset transaction type" in output
    assert "Prepare transaction purpose, necessity, valuation, and expected benefit for review" in output
