"""Hybrid retrieval: BM25 keyword search + dense vectors, merged with Reciprocal Rank Fusion.

Keyword search catches exact terms ("HIPAA", "intake form"); vectors catch paraphrases
("sign-in sheet" vs "check-in"). RRF combines the two rankings without having to tune
score scales against each other.
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

STOPWORDS = frozenset(
    """a an and are as at be been but by can could do does for from had has have he her his i if in into
    is it its me my no not of on or our she so than that the their them then there these they this to
    too up us was we were what when where which who will with would you your just also very really""".split()
)

_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def _stem(word: str) -> str:
    """Tiny suffix stripper: enough to match nurse/nurses, track/tracking, entry/entries."""
    if word.endswith(("sses", "ches", "shes", "xes", "zes")) and len(word) > 5:
        return word[:-2]
    for suffix in ("ations", "ation", "ings", "ing", "ies", "edly", "ed", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3 and not word.endswith("ss"):
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


def tokenize(text: str) -> list[str]:
    return [_stem(w) for w in _WORD.findall(text.lower()) if w not in STOPWORDS and len(w) > 1]


class Embedder(Protocol):
    name: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """Local, deterministic embeddings via feature hashing of unigrams and bigrams.

    No model download and no API cost, so the demo runs anywhere (including air-gapped).
    Swap in OpenAI or Ollama embeddings with ENGINE_EMBEDDER for better semantic recall.
    """

    name = "hashing-512"

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _bucket(self, feature: str) -> tuple[int, float]:
        h = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "big")
        return h % self.dim, (1.0 if (h >> 63) & 1 else -1.0)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out = []
        for text in texts:
            toks = tokenize(text)
            feats = Counter(toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:], strict=False)])
            vec = [0.0] * self.dim
            for feat, tf in feats.items():
                idx, sign = self._bucket(feat)
                vec[idx] += sign * (1.0 + math.log(tf))
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


@dataclass
class Hit:
    chunk_id: str
    filename: str
    text: str
    score: float
    bm25_rank: int | None
    vector_rank: int | None

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class HybridIndex:
    def __init__(self, chunks: list[dict], embedder: Embedder, k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.embedder = embedder
        self.k1, self.b = k1, b
        self.tokens = [tokenize(c["text"]) for c in chunks]
        self.tf = [Counter(t) for t in self.tokens]
        self.avgdl = (sum(len(t) for t in self.tokens) / len(self.tokens)) if self.tokens else 0.0
        df: Counter[str] = Counter()
        for t in self.tokens:
            df.update(set(t))
        n = len(chunks)
        self.idf = {term: math.log(1 + (n - d + 0.5) / (d + 0.5)) for term, d in df.items()}
        self.vectors = embedder.embed([c["text"] for c in chunks]) if chunks else []

    def _bm25(self, query: str) -> list[tuple[int, float]]:
        q = tokenize(query)
        scores = []
        for i, tf in enumerate(self.tf):
            dl = len(self.tokens[i])
            s = 0.0
            for term in q:
                f = tf.get(term)
                if not f:
                    continue
                s += self.idf[term] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1)))
            if s > 0:
                scores.append((i, s))
        return sorted(scores, key=lambda x: -x[1])

    def _dense(self, query: str) -> list[tuple[int, float]]:
        if not self.vectors:
            return []
        qv = self.embedder.embed([query])[0]
        scores = [(i, _cosine(qv, v)) for i, v in enumerate(self.vectors)]
        return sorted((s for s in scores if s[1] > 0.05), key=lambda x: -x[1])

    def search(self, query: str, k: int = 8, rrf_k: int = 60) -> list[Hit]:
        if not self.chunks or not query.strip():
            return []
        bm25 = self._bm25(query)
        dense = self._dense(query)
        fused: dict[int, float] = {}
        b_rank = {i: r for r, (i, _) in enumerate(bm25, 1)}
        v_rank = {i: r for r, (i, _) in enumerate(dense, 1)}
        for ranks in (b_rank, v_rank):
            for i, r in ranks.items():
                fused[i] = fused.get(i, 0.0) + 1.0 / (rrf_k + r)
        top = sorted(fused.items(), key=lambda x: -x[1])[:k]
        best = top[0][1] if top else 1.0
        return [
            Hit(
                chunk_id=self.chunks[i]["id"],
                filename=self.chunks[i]["filename"],
                text=self.chunks[i]["text"],
                score=round(s / best, 3),
                bm25_rank=b_rank.get(i),
                vector_rank=v_rank.get(i),
            )
            for i, s in top
        ]


class IndexCache:
    """Rebuilds a project's index only when its chunks change."""

    def __init__(self) -> None:
        self._cache: dict[str, tuple[tuple, HybridIndex]] = {}
        self._lock = threading.Lock()

    def get(self, project_id: str, chunks: list[dict], embedder: Embedder) -> HybridIndex:
        key = (embedder.name, len(chunks), chunks[-1]["id"] if chunks else None)
        with self._lock:
            cached = self._cache.get(project_id)
            if cached and cached[0] == key:
                return cached[1]
        index = HybridIndex(chunks, embedder)
        with self._lock:
            self._cache[project_id] = (key, index)
        return index
