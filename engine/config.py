"""Settings read from environment variables, with safe defaults for local demos."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache


def _parse_keys(raw: str) -> dict[str, tuple[str, str]]:
    """Parse ENGINE_API_KEYS="alice:approver:key1,bob:editor:key2" into {key: (actor, role)}."""
    keys: dict[str, tuple[str, str]] = {}
    for entry in filter(None, (part.strip() for part in raw.split(","))):
        parts = entry.split(":")
        if len(parts) != 3:
            raise ValueError(f"ENGINE_API_KEYS entry must be actor:role:key, got {entry!r}")
        actor, role, key = parts
        if role not in {"viewer", "editor", "approver"}:
            raise ValueError(f"Unknown role {role!r} in ENGINE_API_KEYS")
        keys[key] = (actor, role)
    return keys


@dataclass(frozen=True)
class Settings:
    db_path: str = "data/engine.db"
    provider: str = "offline"  # offline | openai | anthropic | ollama
    embedder: str = "hashing"  # hashing | openai | ollama
    openai_model: str = "gpt-4o-mini"
    openai_embed_model: str = "text-embedding-3-small"
    anthropic_model: str = "claude-sonnet-5-5"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "gemma3"
    ollama_embed_model: str = "nomic-embed-text"
    max_upload_bytes: int = 5 * 1024 * 1024
    chunk_words: int = 120
    chunk_overlap: int = 30
    context_chunks: int = 14
    max_spec_repairs: int = 2
    api_keys: dict[str, tuple[str, str]] = field(default_factory=dict)

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_keys)


@lru_cache
def get_settings() -> Settings:
    env = os.environ.get
    return Settings(
        db_path=env("ENGINE_DB_PATH", "data/engine.db"),
        provider=env("ENGINE_PROVIDER", "offline").lower(),
        embedder=env("ENGINE_EMBEDDER", "hashing").lower(),
        openai_model=env("OPENAI_MODEL", "gpt-4o-mini"),
        openai_embed_model=env("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
        anthropic_model=env("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
        ollama_url=env("OLLAMA_URL", "http://localhost:11434").rstrip("/"),
        ollama_model=env("OLLAMA_MODEL", "gemma3"),
        ollama_embed_model=env("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        max_upload_bytes=int(env("ENGINE_MAX_UPLOAD_BYTES", str(5 * 1024 * 1024))),
        api_keys=_parse_keys(env("ENGINE_API_KEYS", "")),
    )
