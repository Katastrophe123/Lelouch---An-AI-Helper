"""Text embedders.

Claude has no embeddings endpoint, so embeddings are computed locally:

* ``FastEmbedEmbedder`` runs a small ONNX model (BAAI/bge-small-en-v1.5 by
  default, 384 dimensions) on the CPU. No API key and no per-call cost; the
  model (~130 MB) is downloaded once on first use.
* ``HashingEmbedder`` is a deterministic bag-of-words hash. It needs nothing
  at all, which makes it the test double, and a crude offline fallback.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
from abc import ABC, abstractmethod
from typing import Any

from app.config import Settings
from app.db.models import EMBEDDING_DIM

_TOKEN = re.compile(r"[a-z0-9]+")


class EmbedderUnavailableError(RuntimeError):
    """The embedding model could not be loaded (e.g. first download failed)."""


class Embedder(ABC):
    """Turns text into fixed-width vectors."""

    name: str = "embedder"
    dim: int = EMBEDDING_DIM

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages for storage."""

    async def embed_query(self, text: str) -> list[float]:
        """Embed a search query (some models treat queries differently)."""
        return (await self.embed_documents([text]))[0]


class HashingEmbedder(Embedder):
    """Feature hashing over word unigrams and bigrams, L2-normalised."""

    name = "hashing"

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        tokens = _TOKEN.findall(text.lower())
        features = tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:], strict=False)]
        vector = [0.0] * self.dim
        for feature in features:
            digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(v * v for v in vector))
        return [v / norm for v in vector] if norm else vector

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]


class FastEmbedEmbedder(Embedder):
    """Local ONNX sentence embeddings via ``fastembed``; loaded lazily."""

    name = "fastembed"

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: Any = None
        self._lock = asyncio.Lock()

    async def _load(self) -> Any:
        async with self._lock:
            if self._model is None:
                from fastembed import TextEmbedding  # heavy import, deferred

                try:
                    model = await asyncio.to_thread(TextEmbedding, model_name=self.model_name)
                except Exception as exc:  # noqa: BLE001 - download/runtime failures
                    raise EmbedderUnavailableError(
                        f"Could not load embedding model {self.model_name}. It is downloaded "
                        f"once on first use, so check the internet connection. ({exc})"
                    ) from exc
                probe = await asyncio.to_thread(lambda: list(model.embed(["probe"]))[0])
                self._model = model
                if len(probe) != self.dim:
                    self._model = None
                    raise EmbedderUnavailableError(
                        f"{self.model_name} produces {len(probe)}-d vectors but the database "
                        f"stores {self.dim}-d ones; pick a {self.dim}-d model or migrate."
                    )
        return self._model

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = await self._load()
        vectors = await asyncio.to_thread(lambda: list(model.embed(texts)))
        return [[float(v) for v in vector] for vector in vectors]

    async def embed_query(self, text: str) -> list[float]:
        model = await self._load()
        vectors = await asyncio.to_thread(lambda: list(model.query_embed([text])))
        return [float(v) for v in vectors[0]]


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_backend == "hashing":
        return HashingEmbedder()
    return FastEmbedEmbedder(settings.embedding_model)
