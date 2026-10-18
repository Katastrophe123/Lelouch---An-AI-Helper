"""Service-layer tests: is a turn persisted faithfully, including failures?"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from langgraph.errors import GraphRecursionError

from app.config import Settings
from app.db.models import Conversation
from app.db.repository import ConversationRepository
from app.db.session import Database
from app.models.enums import AnswerStatus, FallbackReason, MessageRole
from app.services.tutor_service import (
    ConversationNotFoundError,
    TutorService,
)
from tests.fakes import (
    EXPLANATION,
    ScriptedLLMClient,
    StreamingScriptedLLMClient,
    build_self_check,
    build_solution,
    build_understanding,
    happy_script,
)

QUESTION = "A ball is dropped from 20 m. How long until it hits the ground?"


class _ExplodingGraph:
    """Graph double whose invocation raises, to test the failure paths."""

    def __init__(self, error: BaseException) -> None:
        self._error = error

    async def ainvoke(self, state: Any, config: Any = None) -> Any:
        raise self._error


def build_service(
    settings: Settings, llm: Any, sleep: Any, *, graph: Any | None = None
) -> TutorService:
    return TutorService(settings=settings, llm=llm, sleep=sleep, graph=graph)


class TestDeriveTitle:
    def test_short_questions_are_used_verbatim(self) -> None:
        assert TutorService.derive_title("What is 2 + 2?") == "What is 2 + 2?"

    def test_whitespace_is_collapsed(self) -> None:
        assert TutorService.derive_title("  what   is\n2 + 2? ") == "what is 2 + 2?"

    def test_long_questions_are_truncated_with_an_ellipsis(self) -> None:
        title = TutorService.derive_title("x " * 200)
        assert len(title) <= 60
        assert title.endswith("…")

    def test_empty_question_falls_back_to_the_default(self) -> None:
        assert TutorService.derive_title("   ") == "New conversation"


class TestSuccessfulTurn:
    @pytest.fixture
    def service(self, settings: Settings, fake_sleep: Any) -> TutorService:
        return build_service(settings, ScriptedLLMClient(happy_script()), fake_sleep)

    async def test_both_messages_are_persisted(
        self,
        service: TutorService,
        conversation: Conversation,
        repository: ConversationRepository,
    ) -> None:
        result = await service.run_turn(
            conversation=conversation, content=QUESTION, repository=repository
        )

        assert result.user_message.role == MessageRole.USER.value
        assert result.user_message.content == QUESTION
        assert result.assistant_message.role == MessageRole.ASSISTANT.value
        assert result.assistant_message.content == EXPLANATION
        assert conversation.message_count == 2

    async def test_run_metadata_is_persisted(
        self,
        service: TutorService,
        conversation: Conversation,
        repository: ConversationRepository,
    ) -> None:
        result = await service.run_turn(
            conversation=conversation, content=QUESTION, repository=repository
        )
        message = result.assistant_message

        assert message.answer_status == AnswerStatus.VERIFIED.value
        assert message.fallback_reason is None
        assert message.confidence == pytest.approx(0.92)
        assert message.retry_count == 0
        assert message.latency_ms is not None and message.latency_ms >= 0
        assert message.verified is True

    async def test_every_node_is_recorded_in_the_trace(
        self,
        service: TutorService,
        conversation: Conversation,
        repository: ConversationRepository,
    ) -> None:
        result = await service.run_turn(
            conversation=conversation, content=QUESTION, repository=repository
        )
        trace = result.assistant_message.trace

        assert [step["node"] for step in trace] == [
            "understand",
            "solve",
            "self_check",
            "explain",
        ]
        assert [step["routing_decision"] for step in trace] == [
            "to_solve",
            "to_self_check",
            "to_explain",
            "end",
        ]

    async def test_subject_is_backfilled_onto_the_conversation(
        self,
        settings: Settings,
        fake_sleep: Any,
        conversation: Conversation,
        repository: ConversationRepository,
    ) -> None:
        llm = ScriptedLLMClient(
            {**happy_script(), "understand": [build_understanding(subject="physics")]}
        )
        service = build_service(settings, llm, fake_sleep)

        await service.run_turn(conversation=conversation, content=QUESTION, repository=repository)

        assert conversation.subject == "physics"

    async def test_title_is_derived_from_the_first_question(
        self,
        service: TutorService,
        repository: ConversationRepository,
        user: Any,
    ) -> None:
        unnamed = await repository.create_conversation(user_id=user.id)
        assert unnamed.title == "New conversation"

        await service.run_turn(conversation=unnamed, content=QUESTION, repository=repository)

        assert unnamed.title == TutorService.derive_title(QUESTION)

    async def test_an_existing_title_is_not_overwritten(
        self,
        service: TutorService,
        conversation: Conversation,
        repository: ConversationRepository,
    ) -> None:
        await service.run_turn(conversation=conversation, content=QUESTION, repository=repository)
        assert conversation.title == "Physics revision"

    async def test_messages_are_eager_loaded_for_serialisation(
        self,
        service: TutorService,
        conversation: Conversation,
        repository: ConversationRepository,
    ) -> None:
        """Touching ``conversation.messages`` after a turn must not lazy-load."""
        await service.run_turn(conversation=conversation, content=QUESTION, repository=repository)
        assert len(conversation.messages) == 2

    async def test_prior_turns_become_model_context(
        self,
        settings: Settings,
        fake_sleep: Any,
        conversation: Conversation,
        repository: ConversationRepository,
    ) -> None:
        llm = ScriptedLLMClient(
            {
                "understand": [build_understanding(), build_understanding()],
                "solve": [build_solution(), build_solution()],
                "self_check": [build_self_check(), build_self_check()],
                "explain": [EXPLANATION, EXPLANATION],
            }
        )
        service = build_service(settings, llm, fake_sleep)

        await service.run_turn(
            conversation=conversation,
            content="First question about free fall?",
            repository=repository,
        )
        await service.run_turn(
            conversation=conversation,
            content="And what about air resistance?",
            repository=repository,
        )

        prompts = llm.prompts_for("understand")
        # Turn 1 reports an empty history; turn 2 replays turn 1's question and
        # answer as dialogue context.
        assert "first question in the conversation" in prompts[0]
        assert "user: First question about free fall?" in prompts[1]
        assert "assistant:" in prompts[1]

    async def test_the_question_appears_once_in_its_own_prompt(
        self,
        settings: Settings,
        fake_sleep: Any,
        conversation: Conversation,
        repository: ConversationRepository,
    ) -> None:
        """History is read before the question is stored, so it is not doubled."""
        llm = ScriptedLLMClient(happy_script())
        service = build_service(settings, llm, fake_sleep)

        await service.run_turn(conversation=conversation, content=QUESTION, repository=repository)

        assert llm.prompts_for("understand")[0].count(QUESTION) == 1


class TestFallbackTurn:
    async def test_unverified_answer_records_its_reason(
        self,
        settings: Settings,
        fake_sleep: Any,
        conversation: Conversation,
        repository: ConversationRepository,
    ) -> None:
        llm = ScriptedLLMClient(
            {
                "understand": [build_understanding()],
                "solve": [build_solution(answer="probably wrong but long enough")],
                "self_check": [build_self_check(passed=False, issues=["Wrong formula."])],
                "explain": [EXPLANATION],
            }
        )
        service = build_service(settings, llm, fake_sleep)

        result = await service.run_turn(
            conversation=conversation, content=QUESTION, repository=repository
        )
        message = result.assistant_message

        assert message.answer_status == AnswerStatus.UNVERIFIED.value
        assert message.fallback_reason == FallbackReason.SELF_CHECK_FAILED.value
        assert message.retry_count == settings.max_solve_retries
        assert message.verified is False
        assert "could not fully verify" in message.content

    async def test_node_error_is_recorded_on_the_message(
        self,
        settings: Settings,
        fake_sleep: Any,
        conversation: Conversation,
        repository: ConversationRepository,
    ) -> None:
        from app.llm.base import LLMTransientError

        llm = ScriptedLLMClient(
            {
                "understand": [LLMTransientError("provider unavailable")],
                "solve": [build_solution()],
                "self_check": [build_self_check()],
                "explain": [EXPLANATION],
            }
        )
        service = build_service(settings, llm, fake_sleep)

        result = await service.run_turn(
            conversation=conversation, content=QUESTION, repository=repository
        )

        assert result.assistant_message.fallback_reason == FallbackReason.NODE_ERROR.value
        assert "provider unavailable" in (result.assistant_message.error or "")


class TestRecursionGuard:
    async def test_recursion_limit_becomes_a_step_limit_fallback(
        self,
        settings: Settings,
        conversation: Conversation,
        repository: ConversationRepository,
    ) -> None:
        service = TutorService(
            settings=settings,
            llm=ScriptedLLMClient(happy_script()),
            graph=_ExplodingGraph(GraphRecursionError("limit of 20 reached")),
        )

        result = await service.run_turn(
            conversation=conversation, content=QUESTION, repository=repository
        )
        message = result.assistant_message

        assert message.answer_status == AnswerStatus.UNVERIFIED.value
        assert message.fallback_reason == FallbackReason.STEP_LIMIT_REACHED.value
        assert "step budget" in message.content

    async def test_unexpected_graph_error_propagates(
        self,
        settings: Settings,
        conversation: Conversation,
        repository: ConversationRepository,
    ) -> None:
        """An unexpected bug must surface as a 500, not a silent bad answer."""
        service = TutorService(
            settings=settings,
            llm=ScriptedLLMClient(happy_script()),
            graph=_ExplodingGraph(RuntimeError("graph blew up")),
        )

        with pytest.raises(RuntimeError, match="graph blew up"):
            await service.run_turn(
                conversation=conversation, content=QUESTION, repository=repository
            )

    def test_recursion_limit_sits_above_the_step_budget(self, settings: Settings) -> None:
        service = TutorService(settings=settings, llm=ScriptedLLMClient(happy_script()))
        assert service.recursion_limit > settings.max_graph_steps


class TestStreamTurn:
    async def test_emits_steps_tokens_and_done(
        self,
        settings: Settings,
        fake_sleep: Any,
        database: Database,
        conversation: Conversation,
        user: Any,
    ) -> None:
        llm = StreamingScriptedLLMClient(happy_script())
        service = build_service(settings, llm, fake_sleep)

        events = [
            event
            async for event in service.stream_turn(
                database=database,
                conversation_id=conversation.id,
                user_id=user.id,
                content=QUESTION,
            )
        ]

        names = [event["event"] for event in events]
        assert names.count("done") == 1
        assert names.index("step") < names.index("token") < names.index("done")
        assert names.count("step") == 4
        assert names.count("token") > 1

        done = events[-1]
        assert done["answer_status"] == AnswerStatus.VERIFIED.value
        assert done["content"] == EXPLANATION
        assert uuid.UUID(done["assistant_message_id"])
        assert llm.stream_calls == 1

    async def test_tokens_reassemble_to_the_stored_answer(
        self,
        settings: Settings,
        fake_sleep: Any,
        database: Database,
        conversation: Conversation,
        user: Any,
    ) -> None:
        service = build_service(settings, StreamingScriptedLLMClient(happy_script()), fake_sleep)

        tokens: list[str] = []
        done: dict[str, Any] = {}
        async for event in service.stream_turn(
            database=database,
            conversation_id=conversation.id,
            user_id=user.id,
            content=QUESTION,
        ):
            if event["event"] == "token":
                tokens.append(event["text"])
            elif event["event"] == "done":
                done = event

        assert "".join(tokens).strip() == done["content"]

    async def test_stream_persists_the_turn(
        self,
        settings: Settings,
        fake_sleep: Any,
        database: Database,
        conversation: Conversation,
        user: Any,
    ) -> None:
        service = build_service(settings, StreamingScriptedLLMClient(happy_script()), fake_sleep)

        async for _ in service.stream_turn(
            database=database,
            conversation_id=conversation.id,
            user_id=user.id,
            content=QUESTION,
        ):
            pass

        async with database.session() as session:
            reloaded = await ConversationRepository(session).get_conversation(conversation.id)
            assert reloaded is not None
            assert reloaded.message_count == 2
            assert [m.role for m in reloaded.messages] == ["user", "assistant"]

    async def test_unknown_conversation_emits_a_not_found_error_event(
        self,
        settings: Settings,
        fake_sleep: Any,
        database: Database,
        user: Any,
    ) -> None:
        service = build_service(settings, StreamingScriptedLLMClient(happy_script()), fake_sleep)

        events = [
            event
            async for event in service.stream_turn(
                database=database,
                conversation_id=uuid.uuid4(),
                user_id=user.id,
                content=QUESTION,
            )
        ]

        assert [event["event"] for event in events] == ["error"]
        assert events[0]["code"] == "not_found"

    async def test_another_users_conversation_is_not_streamable(
        self,
        settings: Settings,
        fake_sleep: Any,
        database: Database,
        conversation: Conversation,
        other_user: Any,
    ) -> None:
        service = build_service(settings, StreamingScriptedLLMClient(happy_script()), fake_sleep)

        events = [
            event
            async for event in service.stream_turn(
                database=database,
                conversation_id=conversation.id,
                user_id=other_user.id,
                content=QUESTION,
            )
        ]

        assert events[0]["event"] == "error"
        assert events[0]["code"] == "not_found"

    async def test_internal_failure_emits_a_sanitised_error_event(
        self,
        settings: Settings,
        fake_sleep: Any,
        database: Database,
        conversation: Conversation,
        user: Any,
    ) -> None:
        """An unexpected exception must not leak its message to the client.

        A ``ValueError`` is not an ``LLMError``, so the retry wrapper lets it
        propagate out of the node and the graph, landing in the worker's
        catch-all.
        """
        llm = StreamingScriptedLLMClient(
            {**happy_script(), "understand": [ValueError("secret internal detail")]}
        )
        service = build_service(settings, llm, fake_sleep)

        events = [
            event
            async for event in service.stream_turn(
                database=database,
                conversation_id=conversation.id,
                user_id=user.id,
                content=QUESTION,
            )
        ]

        assert [event["event"] for event in events] == ["error"]
        assert events[0]["code"] == "internal_error"
        assert "secret internal detail" not in events[0]["message"]

    async def test_fallback_answer_is_still_delivered_over_the_stream(
        self,
        settings: Settings,
        fake_sleep: Any,
        database: Database,
        conversation: Conversation,
        user: Any,
    ) -> None:
        """A fallback is not token-streamed, so `done.content` is authoritative."""
        llm = StreamingScriptedLLMClient(
            {
                "understand": [build_understanding()],
                "solve": [build_solution(answer="long enough but rejected answer")],
                "self_check": [build_self_check(passed=False, issues=["Wrong."])],
                "explain": [EXPLANATION],
            }
        )
        service = build_service(settings, llm, fake_sleep)

        events = [
            event
            async for event in service.stream_turn(
                database=database,
                conversation_id=conversation.id,
                user_id=user.id,
                content=QUESTION,
            )
        ]

        done = events[-1]
        assert done["event"] == "done"
        assert done["answer_status"] == AnswerStatus.UNVERIFIED.value
        assert "could not fully verify" in done["content"]
        assert not [event for event in events if event["event"] == "token"]


class TestConversationNotFoundError:
    def test_carries_the_id(self) -> None:
        missing = uuid.uuid4()
        error = ConversationNotFoundError(missing)
        assert error.conversation_id == missing
        assert str(missing) in str(error)
