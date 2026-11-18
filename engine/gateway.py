"""Model gateway: one interface for cloud and local models.

Engine code asks the gateway for "JSON from this prompt" and never imports a vendor SDK,
so switching OpenAI -> Anthropic -> a local Gemma model on Ollama is a config change.
The `offline` provider uses no model at all (rule-based extraction), which keeps the
public demo free and lets the engine run fully air-gapped.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Protocol

from .config import Settings
from .index import Embedder, HashingEmbedder


class ProviderError(RuntimeError):
    pass


class LLMProvider(Protocol):
    name: str
    model: str

    def complete_json(self, system: str, user: str) -> str: ...


def _post(url: str, headers: dict, payload: dict) -> dict:
    import httpx  # imported lazily so the offline engine also runs in the browser (Pyodide)

    try:
        resp = httpx.post(url, headers=headers, json=payload, timeout=httpx.Timeout(90.0, connect=10.0))
    except httpx.HTTPError as exc:
        raise ProviderError(f"Could not reach {url}: {exc}") from exc
    if resp.status_code >= 400:
        raise ProviderError(f"{url} returned {resp.status_code}: {resp.text[:300]}")
    return resp.json()


class OpenAIProvider:
    name = "openai"

    def __init__(self, settings: Settings) -> None:
        self.model = settings.openai_model
        self.key = os.environ.get("OPENAI_API_KEY", "")
        self.base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")

    def complete_json(self, system: str, user: str) -> str:
        data = _post(
            f"{self.base}/chat/completions",
            {"Authorization": f"Bearer {self.key}"},
            {
                "model": self.model,
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            },
        )
        return data["choices"][0]["message"]["content"]


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, settings: Settings) -> None:
        self.model = settings.anthropic_model
        self.key = os.environ.get("ANTHROPIC_API_KEY", "")

    def complete_json(self, system: str, user: str) -> str:
        if not self.model:
            raise ProviderError("Set ANTHROPIC_MODEL to the Anthropic model id you want to use")
        data = _post(
            "https://api.anthropic.com/v1/messages",
            {"x-api-key": self.key, "anthropic-version": "2023-06-01"},
            {
                "model": self.model,
                "max_tokens": 4096,
                "temperature": 0.1,
                "system": system + "\nRespond with a single JSON object and nothing else.",
                "messages": [{"role": "user", "content": user}],
            },
        )
        return "".join(block.get("text", "") for block in data.get("content", []))


class OllamaProvider:
    """Local models (e.g. Gemma, Llama) for offline or air-gapped deployments."""

    name = "ollama"

    def __init__(self, settings: Settings) -> None:
        self.model = settings.ollama_model
        self.url = settings.ollama_url

    def complete_json(self, system: str, user: str) -> str:
        data = _post(
            f"{self.url}/api/chat",
            {},
            {
                "model": self.model,
                "format": "json",
                "stream": False,
                "options": {"temperature": 0.1},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            },
        )
        return data["message"]["content"]


# Registry so tests (or plugins) can add providers without touching engine code.
PROVIDERS: dict[str, Callable[[Settings], LLMProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "ollama": OllamaProvider,
}


def is_configured(name: str, settings: Settings) -> bool:
    if name == "offline":
        return True
    if name == "openai":
        return bool(os.environ.get("OPENAI_API_KEY"))
    if name == "anthropic":
        return bool(os.environ.get("ANTHROPIC_API_KEY") and settings.anthropic_model)
    if name == "ollama":
        return bool(os.environ.get("OLLAMA_URL"))
    return name in PROVIDERS


def get_provider(name: str, settings: Settings) -> LLMProvider | None:
    """Return the LLM provider, or None for the rule-based offline provider."""
    name = name.lower()
    if name == "offline":
        return None
    if name not in PROVIDERS:
        raise ProviderError(f"Unknown provider {name!r}. Choose from: offline, {', '.join(PROVIDERS)}")
    return PROVIDERS[name](settings)


def list_providers(settings: Settings) -> list[dict]:
    models = {
        "offline": "rule-based extractor",
        "openai": settings.openai_model,
        "anthropic": settings.anthropic_model or "set ANTHROPIC_MODEL",
        "ollama": settings.ollama_model,
    }
    names = ["offline", *PROVIDERS]
    return [
        {
            "name": n,
            "model": models.get(n, "custom"),
            "configured": is_configured(n, settings),
            "default": n == settings.provider,
        }
        for n in names
    ]


# ---------- embedders ----------


class OpenAIEmbedder:
    def __init__(self, settings: Settings) -> None:
        self.model = settings.openai_embed_model
        self.name = f"openai:{self.model}"
        self.key = os.environ.get("OPENAI_API_KEY", "")
        self.base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 96):
            data = _post(
                f"{self.base}/embeddings",
                {"Authorization": f"Bearer {self.key}"},
                {"model": self.model, "input": list(texts[i : i + 96])},
            )
            out.extend(item["embedding"] for item in sorted(data["data"], key=lambda d: d["index"]))
        return out


class OllamaEmbedder:
    def __init__(self, settings: Settings) -> None:
        self.model = settings.ollama_embed_model
        self.name = f"ollama:{self.model}"
        self.url = settings.ollama_url

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        data = _post(f"{self.url}/api/embed", {}, {"model": self.model, "input": list(texts)})
        return data["embeddings"]


def get_embedder(settings: Settings) -> Embedder:
    if settings.embedder == "openai":
        return OpenAIEmbedder(settings)
    if settings.embedder == "ollama":
        return OllamaEmbedder(settings)
    return HashingEmbedder()
