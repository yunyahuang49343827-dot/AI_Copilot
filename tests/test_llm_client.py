import requests
import pytest

from src.generation.llm_client import (
    DEFAULT_TIMEOUT_SECONDS,
    DeepSeekLLMClient,
    LLMConfig,
    LLMConfigurationError,
    LLMGenerationError,
    load_deepseek_config_from_env,
)


class FakeResponse:
    def __init__(self, payload=None, raise_error=None, json_error=None):
        self.payload = payload if payload is not None else {}
        self.raise_error = raise_error
        self.json_error = json_error

    def raise_for_status(self):
        if self.raise_error:
            raise self.raise_error

    def json(self):
        if self.json_error:
            raise self.json_error
        return self.payload


def test_load_config_from_env_and_hides_api_key():
    config = load_deepseek_config_from_env(
        {
            "DEEPSEEK_API_KEY": "secret-key",
            "DEEPSEEK_BASE_URL": "https://deepseek.example/v1",
            "DEEPSEEK_MODEL": "deepseek-test",
            "DEEPSEEK_TIMEOUT_SECONDS": "12.5",
        }
    )

    assert config.api_key == "secret-key"
    assert config.base_url == "https://deepseek.example/v1"
    assert config.model == "deepseek-test"
    assert config.timeout_seconds == 12.5
    assert "secret-key" not in repr(config)


def test_missing_api_key_is_not_configured_and_generate_raises():
    client = DeepSeekLLMClient(LLMConfig(api_key=None))

    assert client.is_configured() is False
    with pytest.raises(LLMConfigurationError):
        client.generate("prompt")


def test_invalid_timeout_env_falls_back_to_default():
    config = load_deepseek_config_from_env({"DEEPSEEK_TIMEOUT_SECONDS": "not-a-number"})

    assert config.timeout_seconds == DEFAULT_TIMEOUT_SECONDS


def test_deepseek_client_builds_expected_request(monkeypatch):
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse({"choices": [{"message": {"content": "Grounded answer [C1]"}}]})

    monkeypatch.setattr(requests, "post", fake_post)
    config = LLMConfig(
        api_key="secret-key",
        base_url="https://api.deepseek.com/v1",
        model="deepseek-test",
        timeout_seconds=9,
    )

    answer = DeepSeekLLMClient(config).generate("Use evidence only.")

    assert answer == "Grounded answer [C1]"
    assert captured["url"] == "https://api.deepseek.com/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer secret-key"
    assert captured["json"]["model"] == "deepseek-test"
    assert captured["json"]["messages"][0]["role"] == "system"
    assert captured["json"]["messages"][1] == {"role": "user", "content": "Use evidence only."}
    assert captured["json"]["temperature"] == 0.0
    assert captured["timeout"] == 9


def test_request_errors_become_generation_error(monkeypatch):
    def fake_post(*_args, **_kwargs):
        raise requests.Timeout("timed out")

    monkeypatch.setattr(requests, "post", fake_post)
    client = DeepSeekLLMClient(LLMConfig(api_key="secret-key"))

    with pytest.raises(LLMGenerationError):
        client.generate("prompt")


def test_malformed_response_becomes_generation_error(monkeypatch):
    def fake_post(*_args, **_kwargs):
        return FakeResponse({"unexpected": []})

    monkeypatch.setattr(requests, "post", fake_post)
    client = DeepSeekLLMClient(LLMConfig(api_key="secret-key"))

    with pytest.raises(LLMGenerationError):
        client.generate("prompt")


def test_non_json_response_becomes_generation_error(monkeypatch):
    def fake_post(*_args, **_kwargs):
        return FakeResponse(json_error=ValueError("bad json"))

    monkeypatch.setattr(requests, "post", fake_post)
    client = DeepSeekLLMClient(LLMConfig(api_key="secret-key"))

    with pytest.raises(LLMGenerationError):
        client.generate("prompt")
