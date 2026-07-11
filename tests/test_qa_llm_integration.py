from backend import services
from backend.schemas import QueryRequest


def sample_retrieved_result(**overrides):
    result = {
        "chunk_id": "TEST-001-article-005",
        "document_id": "WITS-001",
        "policy_name": "防範內線交易管理程序",
        "article": "第五條",
        "article_title": "重大消息",
        "page_start": 1,
        "page_end": 2,
        "source_file": "policy.pdf",
        "text": "公司重大資訊尚未公告前，知悉內部重大資訊之人不得買賣公司股票或有價證券。",
        "text_preview": "公司重大資訊尚未公告前，知悉內部重大資訊之人不得買賣公司股票或有價證券。",
        "final_score": 0.95,
        "keyword_boost": 0.2,
        "policy_boost": 0.2,
        "boilerplate_penalty": 0.0,
    }
    result.update(overrides)
    return result


class FakeLLMClient:
    def __init__(self, answer):
        self.answer = answer
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.answer


class FailingLLMClient:
    def generate(self, _prompt):
        raise RuntimeError("secret-key-value")


def test_query_request_use_llm_defaults_false():
    request = QueryRequest(query="公司還沒公告重大資訊，可以買股票嗎？", top_k=5)

    assert request.use_llm is False


def test_build_qa_response_use_llm_false_does_not_create_client(monkeypatch):
    def fail_create_client():
        raise AssertionError("LLM client should not be created when use_llm is false")

    monkeypatch.setattr(services, "create_deepseek_client_from_env", fail_create_client)

    response = services.build_qa_response(
        "公司還沒公告重大資訊，可以買股票嗎？",
        top_k=5,
        retrieved_results=[sample_retrieved_result()],
        use_llm=False,
    )

    assert response["generation_mode"] == "deterministic"
    assert response["llm_metadata"]["used_llm"] is False
    assert "Based on" in response["answer"] or "retrieved" in response["answer"]


def test_build_qa_response_use_llm_true_with_fake_client_returns_grounded():
    fake_client = FakeLLMClient("Do not trade before announcement under WITS-001 第五條.")

    response = services.build_qa_response(
        "公司還沒公告重大資訊，可以買股票嗎？",
        top_k=5,
        retrieved_results=[sample_retrieved_result()],
        use_llm=True,
        llm_client=fake_client,
    )

    assert response["answer"] == "Do not trade before announcement under WITS-001 第五條."
    assert response["generation_mode"] == "llm_grounded"
    assert response["llm_metadata"]["provider"] == "deepseek"
    assert response["llm_metadata"]["used_llm"] is True
    assert response["llm_metadata"]["evidence_count"] == 1
    assert fake_client.prompts
    assert "Authorization" not in str(response["llm_metadata"])


def test_build_qa_response_llm_error_falls_back_without_secret():
    response = services.build_qa_response(
        "公司還沒公告重大資訊，可以買股票嗎？",
        top_k=5,
        retrieved_results=[sample_retrieved_result()],
        use_llm=True,
        llm_client=FailingLLMClient(),
    )

    assert response["generation_mode"] == "llm_fallback"
    assert response["llm_metadata"]["used_llm"] is False
    assert response["llm_metadata"]["fallback_reason"] == "RuntimeError"
    assert "secret-key-value" not in str(response["llm_metadata"])
    assert "Direct Answer" in response["answer"]


def test_missing_api_key_falls_back_without_real_api_call(monkeypatch):
    class UnconfiguredClient:
        def generate(self, _prompt):
            from src.generation.llm_client import LLMConfigurationError

            raise LLMConfigurationError("DeepSeek API key is not configured.")

    monkeypatch.setattr(services, "create_deepseek_client_from_env", lambda: UnconfiguredClient())

    response = services.build_qa_response(
        "公司還沒公告重大資訊，可以買股票嗎？",
        top_k=5,
        retrieved_results=[sample_retrieved_result()],
        use_llm=True,
    )

    assert response["generation_mode"] == "llm_fallback"
    assert response["llm_metadata"]["fallback_reason"] == "LLMConfigurationError"
    assert response["llm_metadata"]["used_llm"] is False


def test_insufficient_evidence_skips_llm_client_creation(monkeypatch):
    def fail_create_client():
        raise AssertionError("LLM client should not be created for insufficient evidence")

    monkeypatch.setattr(services, "create_deepseek_client_from_env", fail_create_client)
    weak_result = sample_retrieved_result(
        document_id="WITS-999",
        policy_name="無關政策",
        article="第九條",
        article_title="一般事項",
        text="本辦法經董事會通過後實施，修訂時亦同。",
        text_preview="本辦法經董事會通過後實施，修訂時亦同。",
        final_score=0.05,
        keyword_boost=0.0,
        policy_boost=0.0,
        boilerplate_penalty=0.1,
    )

    response = services.build_qa_response(
        "公司員工旅遊補助是多少？",
        top_k=5,
        retrieved_results=[weak_result],
        use_llm=True,
    )

    assert response["generation_mode"] == "llm_fallback"
    assert response["llm_metadata"]["fallback_reason"] == "insufficient_evidence"
    assert response["llm_metadata"]["used_llm"] is False
    assert "insufficient" in response["answer"].lower()


def test_response_metadata_does_not_include_prompt_or_api_key():
    response = services.build_qa_response(
        "公司還沒公告重大資訊，可以買股票嗎？",
        top_k=5,
        retrieved_results=[sample_retrieved_result()],
        use_llm=True,
        llm_client=FakeLLMClient("Do not trade under WITS-001."),
    )

    metadata_text = str(response["llm_metadata"]).lower()
    assert "prompt" not in metadata_text
    assert "api_key" not in metadata_text
    assert "authorization" not in metadata_text
