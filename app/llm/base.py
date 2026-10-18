"""Provider-agnostic LLM client contract.

Every model call goes through :class:`LLMClient`, so the v1 Hugging Face
Llama-3 setup, an OpenAI-compatible endpoint and the offline test stub are
interchangeable by configuration alone.

Beyond v1's single ``invoke``, this contract adds :meth:`LLMClient.stream`,
which the ``explain`` node uses to push tokens to the SSE endpoint.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any


class LLMError(Exception):
    """Base class for all LLM failures."""


class LLMTransientError(LLMError):
    """Failure that is worth retrying (timeout, 5xx, rate limit, bad JSON)."""


class LLMPermanentError(LLMError):
    """Failure that retrying cannot fix (bad credentials, unknown model)."""


class LLMInvalidResponseError(LLMTransientError):
    """The model replied, but not with the structure we asked for."""


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    model: str
    raw: dict[str, Any] = field(default_factory=dict)


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json_object(text: str) -> dict[str, Any]:
    """Best-effort extraction of a single JSON object from model output.

    Handles the three things models actually do: return clean JSON, wrap it in
    a markdown fence, or surround it with prose. Raises
    :class:`LLMInvalidResponseError` when nothing usable is found, which the
    retry wrapper treats as transient.
    """
    candidates: list[str] = []
    stripped = text.strip()
    if stripped:
        candidates.append(stripped)
    fenced = _FENCE_RE.search(text)
    if fenced:
        candidates.append(fenced.group(1))
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise LLMInvalidResponseError(
        f"Expected a JSON object in the model response, got: {text[:300]!r}"
    )


def chunk_text(text: str, size: int = 24) -> list[str]:
    """Split text into word-aligned chunks.

    Used as the streaming fallback for providers without a streaming API, so
    the SSE contract is identical regardless of provider.
    """
    if not text:
        return []
    chunks: list[str] = []
    current = ""
    for word in text.split(" "):
        candidate = f"{current} {word}" if current else word
        if len(candidate) >= size:
            chunks.append(candidate + " ")
            current = ""
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


class LLMClient(ABC):
    """Minimal surface the tutoring agent needs from a language model."""

    name: str = "llm"
    """Provider identifier, e.g. ``huggingface``."""

    model: str = "unknown"
    """Concrete model identifier, surfaced in ``/health`` and logs."""

    supports_streaming: bool = False
    """Whether :meth:`stream` talks to a real streaming API."""

    @abstractmethod
    async def generate(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 900,
    ) -> LLMResponse:
        """Return a completion for the given system/user pair."""

    async def generate_json(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 900,
    ) -> dict[str, Any]:
        """Call :meth:`generate` and parse the reply as a JSON object."""
        response = await self.generate(
            system=system, user=user, temperature=temperature, max_tokens=max_tokens
        )
        return extract_json_object(response.text)

    async def stream(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 900,
    ) -> AsyncIterator[str]:
        """Yield text chunks.

        The default implementation generates the full response and then chunks
        it, so callers get one interface whether or not the provider streams.
        """
        response = await self.generate(
            system=system, user=user, temperature=temperature, max_tokens=max_tokens
        )
        for piece in chunk_text(response.text):
            yield piece

    async def aclose(self) -> None:
        """Release any underlying network resources."""
        return
