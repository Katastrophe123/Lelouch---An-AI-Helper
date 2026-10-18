"""Tests for the LLM layer: contract, retries, providers and the factory."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from app.config import Settings
from app.llm.base import (
    LLMInvalidResponseError,
    LLMPermanentError,
    LLMTransientError,
    chunk_text,
    extract_json_object,
)
from app.llm.factory import build_llm_client
from app.llm.providers import (
    HuggingFaceInferenceClient,
    OpenAICompatibleClient,
    StubLLMClient,
    evaluate_simple_arithmetic,
)
from app.llm.retry import RetriesExhaustedError, call_with_retries
from tests.fakes import ScriptedLLMClient


class TestExtractJsonObject:
    def test_parses_clean_json(self) -> None:
        assert extract_json_object('{"subject": "math"}') == {"subject": "math"}

    def test_parses_fenced_json(self) -> None:
        raw = 'Sure!\n```json\n{"passed": true, "confidence": 0.8}\n```\n'
        assert extract_json_object(raw) == {"passed": True, "confidence": 0.8}

    def test_parses_json_surrounded_by_prose(self) -> None:
        raw = 'Here you go: {"passed": false} - hope that helps.'
        assert extract_json_object(raw) == {"passed": False}

    @pytest.mark.parametrize("raw", ["", "no json here", "[1, 2, 3]", '{"unterminated": '])
    def test_rejects_unusable_payloads(self, raw: str) -> None:
        with pytest.raises(LLMInvalidResponseError):
            extract_json_object(raw)


class TestChunkText:
    def test_empty_text_yields_nothing(self) -> None:
        assert chunk_text("") == []

    def test_chunks_reassemble_to_the_original(self) -> None:
        text = "The quick brown fox jumps over the lazy dog repeatedly and often."
        assert "".join(chunk_text(text, size=10)).strip() == text

    def test_short_text_is_a_single_chunk(self) -> None:
        assert chunk_text("hi", size=50) == ["hi"]

    def test_chunk_count_grows_as_size_shrinks(self) -> None:
        text = "one two three four five six seven eight nine ten"
        assert len(chunk_text(text, size=5)) > len(chunk_text(text, size=40))


class TestDefaultStreaming:
    async def test_base_stream_chunks_a_non_streaming_provider(self) -> None:
        """A provider without streaming still satisfies the stream contract."""
        client = ScriptedLLMClient({"explain": ["a b c d e f g h i j k l m n o p"]})
        pieces = [piece async for piece in client.stream(system="TASK: explain", user="x")]

        assert len(pieces) > 1
        assert "".join(pieces).strip() == "a b c d e f g h i j k l m n o p"


class TestCallWithRetries:
    async def test_returns_immediately_on_success(
        self, fake_sleep: Any, sleep_calls: list[float]
    ) -> None:
        async def operation() -> str:
            return "ok"

        assert await call_with_retries(operation, max_attempts=3, sleep=fake_sleep) == "ok"
        assert sleep_calls == []

    async def test_backoff_is_exponential(self, fake_sleep: Any, sleep_calls: list[float]) -> None:
        async def operation() -> str:
            raise LLMTransientError("always fails")

        with pytest.raises(RetriesExhaustedError):
            await call_with_retries(operation, max_attempts=4, base_delay=0.5, sleep=fake_sleep)

        # One sleep fewer than attempts: no delay after the final failure.
        assert sleep_calls == [0.5, 1.0, 2.0]

    async def test_permanent_errors_are_not_retried(
        self, fake_sleep: Any, sleep_calls: list[float]
    ) -> None:
        attempts = 0

        async def operation() -> str:
            nonlocal attempts
            attempts += 1
            raise LLMPermanentError("bad credentials")

        with pytest.raises(LLMPermanentError):
            await call_with_retries(operation, max_attempts=5, sleep=fake_sleep)

        assert attempts == 1
        assert sleep_calls == []

    async def test_retries_exhausted_is_a_permanent_error(self, fake_sleep: Any) -> None:
        async def operation() -> str:
            raise LLMTransientError("down")

        with pytest.raises(RetriesExhaustedError) as excinfo:
            await call_with_retries(operation, max_attempts=2, sleep=fake_sleep)

        assert excinfo.value.attempts == 2
        assert isinstance(excinfo.value, LLMPermanentError)

    async def test_timeouts_count_as_transient(self, fake_sleep: Any) -> None:
        attempts = 0

        async def operation() -> str:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise httpx.TimeoutException("timed out")
            return "ok"

        assert await call_with_retries(operation, max_attempts=2, sleep=fake_sleep) == "ok"

    async def test_unexpected_exceptions_propagate_unwrapped(self, fake_sleep: Any) -> None:
        async def operation() -> str:
            raise ValueError("programmer error")

        with pytest.raises(ValueError, match="programmer error"):
            await call_with_retries(operation, max_attempts=3, sleep=fake_sleep)


def _openai_client(transport: httpx.MockTransport) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        api_key="test-key",
        base_url="https://llm.example/v1",
        model="test-model",
        client=httpx.AsyncClient(transport=transport, base_url="https://llm.example/v1"),
    )


class TestOpenAICompatibleClient:
    async def test_returns_message_content(self) -> None:
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "model": "test-model",
                    "choices": [{"message": {"content": "hello there"}}],
                },
            )

        client = _openai_client(httpx.MockTransport(handler))
        response = await client.generate(system="TASK: explain", user="hi", max_tokens=42)

        assert response.text == "hello there"
        assert captured["body"]["max_tokens"] == 42
        assert captured["body"]["stream"] is False

    @pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
    async def test_retryable_statuses_raise_transient(self, status: int) -> None:
        client = _openai_client(
            httpx.MockTransport(lambda request: httpx.Response(status, text="busy"))
        )
        with pytest.raises(LLMTransientError):
            await client.generate(system="TASK: explain", user="hi")

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    async def test_client_errors_raise_permanent(self, status: int) -> None:
        client = _openai_client(
            httpx.MockTransport(lambda request: httpx.Response(status, text="nope"))
        )
        with pytest.raises(LLMPermanentError):
            await client.generate(system="TASK: explain", user="hi")

    async def test_malformed_success_body_is_transient(self) -> None:
        client = _openai_client(
            httpx.MockTransport(lambda request: httpx.Response(200, json={"unexpected": 1}))
        )
        with pytest.raises(LLMTransientError):
            await client.generate(system="TASK: explain", user="hi")

    async def test_streaming_parses_sse_deltas(self) -> None:
        frames = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            'data: {"choices":[{"delta":{"content":" world"}}]}',
            'data: {"choices":[{"delta":{}}]}',  # no content, skipped
            "data: [DONE]",
            "",
        ]
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, text="\n\n".join(frames))

        client = _openai_client(httpx.MockTransport(handler))
        pieces = [piece async for piece in client.stream(system="TASK: explain", user="hi")]

        assert pieces == ["Hello", " world"]
        assert captured["body"]["stream"] is True
        assert client.supports_streaming is True

    async def test_streaming_skips_a_malformed_frame(self) -> None:
        """A bad frame mid-stream must not abort a partly-delivered answer."""
        frames = [
            'data: {"choices":[{"delta":{"content":"good"}}]}',
            "data: {not json at all",
            'data: {"choices":[{"delta":{"content":"-ish"}}]}',
            "data: [DONE]",
            "",
        ]
        client = _openai_client(
            httpx.MockTransport(lambda request: httpx.Response(200, text="\n\n".join(frames)))
        )
        pieces = [piece async for piece in client.stream(system="TASK: explain", user="hi")]
        assert pieces == ["good", "-ish"]

    async def test_streaming_error_status_raises(self) -> None:
        client = _openai_client(
            httpx.MockTransport(lambda request: httpx.Response(401, text="bad key"))
        )
        with pytest.raises(LLMPermanentError):
            [piece async for piece in client.stream(system="TASK: explain", user="hi")]


class TestHuggingFaceInferenceClient:
    def _client(self, transport: httpx.MockTransport) -> HuggingFaceInferenceClient:
        return HuggingFaceInferenceClient(
            api_key="hf-key",
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            client=httpx.AsyncClient(
                transport=transport, base_url="https://api-inference.huggingface.co/models"
            ),
        )

    async def test_parses_list_response(self) -> None:
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["path"] = request.url.path
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=[{"generated_text": "llama says hi"}])

        response = await self._client(httpx.MockTransport(handler)).generate(
            system="TASK: explain", user="hi"
        )

        assert response.text == "llama says hi"
        assert "Meta-Llama-3-8B-Instruct" in str(captured["path"])
        # HF rejects temperature == 0; the client must nudge it upwards.
        assert captured["body"]["parameters"]["temperature"] > 0

    def test_prompt_uses_the_llama3_chat_template(self) -> None:
        prompt = HuggingFaceInferenceClient.build_prompt("SYS", "USER")
        assert prompt.startswith("<|begin_of_text|>")
        assert "<|start_header_id|>system<|end_header_id|>" in prompt
        assert "SYS" in prompt and "USER" in prompt
        assert prompt.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")

    async def test_model_loading_503_is_transient(self) -> None:
        client = self._client(
            httpx.MockTransport(lambda request: httpx.Response(503, text="loading"))
        )
        with pytest.raises(LLMTransientError):
            await client.generate(system="TASK: explain", user="hi")

    async def test_streaming_falls_back_to_chunking(self) -> None:
        client = self._client(
            httpx.MockTransport(
                lambda request: httpx.Response(
                    200, json=[{"generated_text": "one two three four five six seven"}]
                )
            )
        )
        pieces = [piece async for piece in client.stream(system="TASK: explain", user="hi")]

        assert client.supports_streaming is False
        assert len(pieces) > 1
        assert "".join(pieces).strip() == "one two three four five six seven"


class TestEvaluateSimpleArithmetic:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("What is 17 * 23?", "391"),
            ("Compute 100 / 4", "25"),
            ("What is 7 + 8?", "15"),
            ("Work out 10 - 25", "-15"),
            ("Evaluate 2 ^ 10", "1024"),
            ("What is 12 x 12?", "144"),
            ("What is 15% of 240?", "36"),
        ],
    )
    def test_evaluates_supported_expressions(self, text: str, expected: str) -> None:
        result = evaluate_simple_arithmetic(text)
        assert result is not None
        assert result[1] == expected

    @pytest.mark.parametrize(
        "text", ["Explain photosynthesis", "What is the capital of France?", ""]
    )
    def test_returns_none_without_an_expression(self, text: str) -> None:
        assert evaluate_simple_arithmetic(text) is None

    def test_division_by_zero_is_not_an_error(self) -> None:
        assert evaluate_simple_arithmetic("What is 5 / 0?") is None

    def test_non_integer_results_keep_decimals(self) -> None:
        result = evaluate_simple_arithmetic("What is 1 / 8?")
        assert result is not None
        assert result[1] == "0.125"


class TestStubLLMClient:
    async def test_understand_returns_valid_json(self) -> None:
        stub = StubLLMClient()
        payload = await stub.generate_json(
            system="TASK: understand",
            user="<question>A ball is dropped. What is the velocity?</question>",
        )
        assert payload["subject"] == "physics"
        assert payload["is_answerable"] is True

    async def test_very_short_question_is_not_answerable(self) -> None:
        stub = StubLLMClient()
        payload = await stub.generate_json(
            system="TASK: understand", user="<question>ok</question>"
        )
        assert payload["is_answerable"] is False

    async def test_solve_actually_computes_arithmetic(self) -> None:
        stub = StubLLMClient()
        payload = await stub.generate_json(
            system="TASK: solve", user="<question>What is 17 * 23?</question>"
        )
        assert "391" in payload["answer"]
        assert payload["steps"]

    async def test_solve_falls_back_to_a_method_outline(self) -> None:
        stub = StubLLMClient()
        payload = await stub.generate_json(
            system="TASK: solve", user="<question>Explain photosynthesis</question>"
        )
        assert payload["answer"]
        assert len(payload["steps"]) >= 3

    async def test_self_check_passes_a_reasonable_answer(self) -> None:
        stub = StubLLMClient()
        payload = await stub.generate_json(
            system="TASK: self_check", user="<answer>17 * 23 = 391</answer>"
        )
        assert payload["passed"] is True
        assert payload["confidence"] >= 0.6

    async def test_forced_failures_drive_the_retry_branch(self) -> None:
        stub = StubLLMClient(force_check_failures=2)
        first = await stub.generate_json(
            system="TASK: self_check", user="<answer>17 * 23 = 391</answer>"
        )
        second = await stub.generate_json(
            system="TASK: self_check", user="<answer>17 * 23 = 391</answer>"
        )
        third = await stub.generate_json(
            system="TASK: self_check", user="<answer>17 * 23 = 391</answer>"
        )

        assert first["passed"] is False
        assert second["passed"] is False
        assert third["passed"] is True

    async def test_explain_renders_numbered_steps(self) -> None:
        stub = StubLLMClient()
        response = await stub.generate(
            system="TASK: explain",
            user="<answer>391</answer><steps>\n- one\n- two\n</steps>",
        )
        assert "**Step 1.**" in response.text
        assert "**Step 2.**" in response.text
        assert "391" in response.text

    async def test_unknown_task_is_permanent(self) -> None:
        with pytest.raises(LLMPermanentError):
            await StubLLMClient().generate(system="TASK: translate", user="hi")


class TestFactory:
    def test_stub_requires_no_api_key(self) -> None:
        assert isinstance(
            build_llm_client(Settings(llm_provider="stub", llm_api_key=None)), StubLLMClient
        )

    def test_openai_selected_by_provider(self) -> None:
        client = build_llm_client(
            Settings(llm_provider="openai", llm_api_key="sk-test", llm_model="gpt-4o-mini")
        )
        assert isinstance(client, OpenAICompatibleClient)
        assert client.model == "gpt-4o-mini"

    def test_huggingface_is_the_v1_model_path(self) -> None:
        client = build_llm_client(
            Settings(
                llm_provider="huggingface",
                llm_api_key="hf-test",
                llm_model="meta-llama/Meta-Llama-3-8B-Instruct",
            )
        )
        assert isinstance(client, HuggingFaceInferenceClient)
        assert client.model == "meta-llama/Meta-Llama-3-8B-Instruct"

    @pytest.mark.parametrize("provider", ["openai", "huggingface"])
    def test_networked_provider_without_key_fails_fast(self, provider: str) -> None:
        with pytest.raises(LLMPermanentError, match="LLM_API_KEY"):
            build_llm_client(Settings(llm_provider=provider, llm_api_key=None))
