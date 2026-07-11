"""Optional DeepSeek LLM client abstraction for grounded generation."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any, Mapping, Protocol

import requests


DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
DEFAULT_TIMEOUT_SECONDS = 30.0

SYSTEM_MESSAGE = (
    "You are a grounded compliance Q&A assistant. Use only the user-provided "
    "evidence in the prompt. Do not call tools, create cases, approve actions, "
    "or perform external actions."
)


class LLMConfigurationError(RuntimeError):
    """Raised when the optional LLM client is not configured."""


class LLMGenerationError(RuntimeError):
    """Raised when the optional LLM provider cannot return a safe response."""


@dataclass(frozen=True)
class LLMConfig:
    api_key: str | None = field(default=None, repr=False)
    base_url: str = DEFAULT_DEEPSEEK_BASE_URL
    model: str = DEFAULT_DEEPSEEK_MODEL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    def is_configured(self) -> bool:
        return bool((self.api_key or "").strip())


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str:
        """Generate a response from a fully prepared prompt."""


def _coerce_timeout(value: str | None) -> float:
    if not value:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        timeout = float(value)
    except ValueError:
        return DEFAULT_TIMEOUT_SECONDS
    if timeout <= 0:
        return DEFAULT_TIMEOUT_SECONDS
    return timeout


def load_deepseek_config_from_env(env: Mapping[str, str] | None = None) -> LLMConfig:
    source = env if env is not None else os.environ
    api_key = (source.get("DEEPSEEK_API_KEY") or "").strip() or None
    base_url = (source.get("DEEPSEEK_BASE_URL") or DEFAULT_DEEPSEEK_BASE_URL).strip()
    model = (source.get("DEEPSEEK_MODEL") or DEFAULT_DEEPSEEK_MODEL).strip()
    timeout_seconds = _coerce_timeout(source.get("DEEPSEEK_TIMEOUT_SECONDS"))
    return LLMConfig(
        api_key=api_key,
        base_url=base_url or DEFAULT_DEEPSEEK_BASE_URL,
        model=model or DEFAULT_DEEPSEEK_MODEL,
        timeout_seconds=timeout_seconds,
    )


class DeepSeekLLMClient:
    def __init__(self, config: LLMConfig, session: Any | None = None):
        self.config = config
        self.session = session or requests

    def is_configured(self) -> bool:
        return self.config.is_configured()

    def _chat_completions_url(self) -> str:
        base_url = self.config.base_url.rstrip("/")
        if base_url.endswith("/chat/completions"):
            return base_url
        return f"{base_url}/chat/completions"

    def generate(self, prompt: str) -> str:
        if not self.is_configured():
            raise LLMConfigurationError("DeepSeek API key is not configured.")

        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
        }
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = self.session.post(
                self._chat_completions_url(),
                headers=headers,
                json=payload,
                timeout=self.config.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            raise LLMGenerationError("DeepSeek request failed.") from exc
        except ValueError as exc:
            raise LLMGenerationError("DeepSeek response was not valid JSON.") from exc

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMGenerationError("DeepSeek response did not contain message content.") from exc

        if not isinstance(content, str) or not content.strip():
            raise LLMGenerationError("DeepSeek response content was empty.")
        return content.strip()


def create_deepseek_client_from_env() -> DeepSeekLLMClient:
    return DeepSeekLLMClient(load_deepseek_config_from_env())
