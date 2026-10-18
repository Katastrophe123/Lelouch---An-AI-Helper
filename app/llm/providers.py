"""Concrete :class:`~app.llm.base.LLMClient` implementations."""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.llm.base import (
    LLMClient,
    LLMPermanentError,
    LLMResponse,
    LLMTransientError,
)
from app.logging_config import get_logger

logger = get_logger(__name__)

_TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def _classify_status(
    status: int, body: str, provider: str
) -> LLMTransientError | LLMPermanentError:
    """Map an HTTP status onto a transient/permanent LLM error."""
    message = f"{provider} returned HTTP {status}: {body[:300]}"
    if status in _TRANSIENT_STATUS:
        return LLMTransientError(message)
    return LLMPermanentError(message)


class OpenAICompatibleClient(LLMClient):
    """Client for any ``/chat/completions`` endpoint (OpenAI, vLLM, Groq, ...).

    This is the only provider with native streaming; it parses the SSE
    ``data:`` frames the Chat Completions API emits when ``stream=true``.
    """

    name = "openai"
    supports_streaming = True

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.model = model
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    def _payload(
        self, system: str, user: str, temperature: float, max_tokens: int, *, stream: bool
    ) -> dict[str, Any]:
        return {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

    async def generate(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 900,
    ) -> LLMResponse:
        response = await self._client.post(
            "/chat/completions",
            json=self._payload(system, user, temperature, max_tokens, stream=False),
        )
        if response.status_code >= 400:
            raise _classify_status(response.status_code, response.text, self.name)

        try:
            data: dict[str, Any] = response.json()
            text = data["choices"][0]["message"]["content"] or ""
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise LLMTransientError(f"Malformed response from {self.name}: {exc}") from exc

        return LLMResponse(text=text, model=data.get("model", self.model), raw=data)

    async def stream(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 900,
    ) -> AsyncIterator[str]:
        payload = self._payload(system, user, temperature, max_tokens, stream=True)
        async with self._client.stream("POST", "/chat/completions", json=payload) as response:
            if response.status_code >= 400:
                body = (await response.aread()).decode(errors="replace")
                raise _classify_status(response.status_code, body, self.name)

            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"].get("content")
                except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                    # A malformed frame mid-stream is skipped rather than
                    # aborting a partially-delivered answer.
                    logger.warning("stream_frame_skipped", provider=self.name)
                    continue
                if delta:
                    yield delta

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


class HuggingFaceInferenceClient(LLMClient):
    """Client for the Hugging Face Inference API text-generation task.

    This is the provider v1 used (``Meta-Llama-3-8B-Instruct``). It keeps the
    Llama-3 chat template that LangChain applied in v1. Streaming falls back to
    the base class's chunking, because the Inference API's streaming shape
    varies by endpoint type and correctness matters more here than token-level
    latency.
    """

    name = "huggingface"
    supports_streaming = False

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://api-inference.huggingface.co/models",
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.model = model
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    @staticmethod
    def build_prompt(system: str, user: str) -> str:
        """Render the Llama-3 instruct chat template."""
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    async def generate(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 900,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "inputs": self.build_prompt(system, user),
            "parameters": {
                "max_new_tokens": max_tokens,
                # HF rejects temperature == 0; nudge to a near-greedy value.
                "temperature": max(temperature, 0.01),
                "return_full_text": False,
            },
            "options": {"wait_for_model": True},
        }
        response = await self._client.post(f"/{self.model}", json=payload)
        if response.status_code >= 400:
            raise _classify_status(response.status_code, response.text, self.name)

        try:
            data = response.json()
            text = data[0]["generated_text"] if isinstance(data, list) else data["generated_text"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise LLMTransientError(f"Malformed response from {self.name}: {exc}") from exc

        return LLMResponse(
            text=text,
            model=self.model,
            raw=data if isinstance(data, dict) else {"data": data},
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


# --------------------------------------------------------------------------- #
# Offline stub
# --------------------------------------------------------------------------- #
_SUBJECT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "math": (
        "integral",
        "derivative",
        "equation",
        "solve for",
        "factorise",
        "factorize",
        "polynomial",
        "matrix",
        "probability",
        "triangle",
        "hypotenuse",
        "percent",
        "fraction",
        "algebra",
        "calculus",
        "geometry",
        "sum of",
    ),
    "physics": (
        "velocity",
        "acceleration",
        "force",
        "newton",
        "momentum",
        "kinetic",
        "friction",
        "gravity",
        "dropped from",
        "joule",
        "watt",
        "circuit",
        "voltage",
    ),
    "chemistry": (
        "mole",
        "molar",
        "ph of",
        "titration",
        "oxidation",
        "covalent",
        "ionic",
        "stoichiometry",
        "balance the equation",
        "periodic table",
    ),
    "biology": (
        "mitosis",
        "photosynthesis",
        "enzyme",
        "dna",
        "chromosome",
        "osmosis",
        "respiration",
        "cell membrane",
        "ecosystem",
    ),
    "computer_science": (
        "algorithm",
        "big o",
        "complexity",
        "recursion",
        "linked list",
        "binary tree",
        "sorting",
        "compile",
        "python",
        "sql query",
        "pointer",
    ),
    "economics": ("elasticity", "gdp", "inflation", "supply and demand", "opportunity cost"),
    "history": ("treaty", "revolution", "world war", "dynasty", "colonial", "century"),
    "geography": ("tectonic", "monsoon", "latitude", "erosion", "climate zone"),
    "english": ("metaphor", "thesis statement", "paragraph", "poem", "grammar", "essay"),
}

_DIFFICULTY_MARKERS: dict[str, tuple[str, ...]] = {
    "university": ("prove that", "derive", "eigen", "laplace", "lagrangian", "asymptotic"),
    "introductory": ("what is", "define", "name the", "list the"),
}

_ARITHMETIC_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*([+\-*/x×^])\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)", re.IGNORECASE)


def evaluate_simple_arithmetic(text: str) -> tuple[str, str] | None:
    """Evaluate a single binary expression or ``X% of Y``.

    Returns ``(expression, result)`` or ``None``. Deliberately pattern-based
    rather than ``eval``: no user string is ever executed.
    """
    percent = _PERCENT_RE.search(text)
    if percent:
        pct, whole = float(percent.group(1)), float(percent.group(2))
        return f"{percent.group(1)}% of {percent.group(2)}", _format_number(pct / 100 * whole)

    match = _ARITHMETIC_RE.search(text)
    if not match:
        return None
    left, operator, right = float(match.group(1)), match.group(2).lower(), float(match.group(3))
    try:
        if operator == "+":
            value = left + right
        elif operator == "-":
            value = left - right
        elif operator in {"*", "x", "×"}:
            value = left * right
        elif operator == "/":
            value = left / right
        else:  # "^"
            value = left**right
    except (ZeroDivisionError, OverflowError):
        return None
    return match.group(0), _format_number(value)


def _format_number(value: float) -> str:
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.4f}".rstrip("0").rstrip(".")


class StubLLMClient(LLMClient):
    """Deterministic, offline, rule-driven stand-in for Llama-3.

    It exists so ``docker compose up`` and the test suite work with no API key.
    It is **not** a quality baseline for the real model - it only demonstrates
    the graph's control flow, including the retry and fallback branches.

    ``force_check_failures`` makes the self-check reject the first *n* answers,
    which is how the retry path is demonstrated without a real model.
    """

    name = "stub"
    supports_streaming = False

    def __init__(self, *, model: str = "stub-tutor-v1", force_check_failures: int = 0) -> None:
        self.model = model
        self._remaining_forced_failures = force_check_failures

    async def generate(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 900,
    ) -> LLMResponse:
        task = self._detect_task(system)
        handler = {
            "understand": self._understand,
            "solve": self._solve,
            "self_check": self._self_check,
            "explain": self._explain,
        }.get(task)
        if handler is None:
            raise LLMPermanentError(f"StubLLMClient received an unknown task: {task!r}")
        return LLMResponse(text=handler(user), model=self.model, raw={"task": task})

    @staticmethod
    def _detect_task(system: str) -> str:
        for line in system.splitlines():
            if line.strip().upper().startswith("TASK:"):
                return line.split(":", 1)[1].strip().lower()
        return "unknown"

    # --- task handlers ----------------------------------------------------
    @staticmethod
    def _understand(user: str) -> str:
        text = user.lower()
        subject = "other"
        for candidate, keywords in _SUBJECT_KEYWORDS.items():
            if any(keyword in text for keyword in keywords):
                subject = candidate
                break
        if subject == "other" and _ARITHMETIC_RE.search(text):
            subject = "math"

        difficulty = "advanced_school"
        for level, markers in _DIFFICULTY_MARKERS.items():
            if any(marker in text for marker in markers):
                difficulty = level
                break

        question = _extract_tagged(user, "question").strip() or user.strip()
        # A question with no interrogative content and almost no text is not
        # something a tutor can act on - this drives the not_answerable branch.
        is_answerable = len(question) >= 12 and (
            "?" in question
            or any(
                word in text
                for word in ("solve", "find", "calculate", "explain", "what", "why", "how", "prove")
            )
        )

        return json.dumps(
            {
                "restated_question": question[:400],
                "subject": subject,
                "topic": subject.replace("_", " ") if subject != "other" else "general",
                "difficulty": difficulty,
                "knowns": _extract_numbers(question)[:4],
                "unknowns": ["the requested value"] if is_answerable else ["the actual question"],
                "is_answerable": is_answerable,
            }
        )

    @staticmethod
    def _solve(user: str) -> str:
        question = _extract_tagged(user, "question").strip() or user.strip()
        computed = evaluate_simple_arithmetic(question)
        if computed:
            expression, result = computed
            return json.dumps(
                {
                    "answer": f"{expression} = {result}",
                    "steps": [
                        f"Identify the operation in '{expression}'.",
                        "Apply it to the two operands.",
                        f"The result is {result}.",
                    ],
                    "assumptions": ["The expression is evaluated with standard precedence."],
                }
            )
        return json.dumps(
            {
                "answer": (
                    "Work through the problem by naming the quantities you know, choosing the "
                    "relationship that connects them, and substituting the known values."
                ),
                "steps": [
                    "List every quantity the question gives you.",
                    "Pick the formula or rule that links those quantities to the unknown.",
                    "Substitute the known values and simplify.",
                    "Sanity-check the magnitude and units of the result.",
                ],
                "assumptions": ["Standard school-level conventions apply."],
            }
        )

    def _self_check(self, user: str) -> str:
        if self._remaining_forced_failures > 0:
            self._remaining_forced_failures -= 1
            return json.dumps(
                {
                    "passed": False,
                    "confidence": 0.25,
                    "issues": ["The working skips a step and the result is not justified."],
                }
            )
        answer = _extract_tagged(user, "answer").strip()
        issues: list[str] = []
        if len(answer) < 10:
            issues.append("The answer is too short to be checked.")
        if "i don't know" in answer.lower() or "cannot" in answer.lower():
            issues.append("The answer does not actually resolve the question.")
        return json.dumps(
            {
                "passed": not issues,
                "confidence": 0.88 if not issues else 0.2,
                "issues": issues,
            }
        )

    @staticmethod
    def _explain(user: str) -> str:
        answer = _extract_tagged(user, "answer").strip()
        steps = [
            line.strip("-• ").strip()
            for line in _extract_tagged(user, "steps").splitlines()
            if line.strip("-• ").strip()
        ]
        lines = ["Here is the reasoning, one step at a time.", ""]
        lines.extend(
            f"**Step {index}.** {step}"
            for index, step in enumerate(steps or ["Restate the problem."], 1)
        )
        lines.extend(
            [
                "",
                f"**Answer.** {answer or 'See the steps above.'}",
                "",
                "Try the same method on a similar problem to check that it has stuck.",
            ]
        )
        return "\n".join(lines)


def _extract_tagged(text: str, tag: str) -> str:
    """Return the content between ``<tag>`` and ``</tag>``, or ``''``."""
    open_tag, close_tag = f"<{tag}>", f"</{tag}>"
    start = text.find(open_tag)
    end = text.find(close_tag, start + len(open_tag))
    if start == -1 or end == -1:
        return ""
    return text[start + len(open_tag) : end]


def _extract_numbers(text: str) -> list[str]:
    return re.findall(r"-?\d+(?:\.\d+)?\s*[a-zA-Z/]*", text)
