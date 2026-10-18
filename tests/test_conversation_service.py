"""Read-side service tests: schema mapping, trace decoding, pagination maths."""

from __future__ import annotations

import uuid

import pytest

from app.db.models import Conversation, User
from app.db.repository import ConversationRepository
from app.models.enums import (
    AnswerStatus,
    ConversationStatus,
    FallbackReason,
    MessageRole,
    Subject,
)
from app.services.conversation_service import (
    ConversationService,
    conversation_to_detail,
    message_to_response,
)
from app.services.tutor_service import ConversationNotFoundError
from tests.test_repository import make_trace


class TestCreate:
    async def test_returns_an_empty_detail(
        self, conversation_service: ConversationService, user: User
    ) -> None:
        detail = await conversation_service.create(
            user_id=user.id, title="Calculus", subject=Subject.MATH
        )

        assert detail.title == "Calculus"
        assert detail.subject is Subject.MATH
        assert detail.status is ConversationStatus.ACTIVE
        assert detail.messages == []


class TestGet:
    async def test_maps_messages_and_decodes_the_trace(
        self,
        conversation_service: ConversationService,
        repository: ConversationRepository,
        conversation: Conversation,
        user: User,
    ) -> None:
        await repository.add_user_message(conversation=conversation, content="q")
        await repository.add_assistant_message(
            conversation=conversation,
            content="a",
            answer_status=AnswerStatus.UNVERIFIED,
            fallback_reason=FallbackReason.SELF_CHECK_FAILED,
            confidence=0.4,
            retry_count=2,
            latency_ms=99,
            trace=make_trace(),
        )

        detail = await conversation_service.get(conversation.id, user_id=user.id)

        assert [m.role for m in detail.messages] == [
            MessageRole.USER,
            MessageRole.ASSISTANT,
        ]
        assistant = detail.messages[1]
        assert assistant.answer_status is AnswerStatus.UNVERIFIED
        assert assistant.fallback_reason is FallbackReason.SELF_CHECK_FAILED
        assert assistant.confidence == pytest.approx(0.4)
        assert [step.node.value for step in assistant.trace] == ["understand", "fallback"]
        assert assistant.trace[0].routing_decision is not None

    async def test_user_messages_have_no_run_metadata(
        self,
        conversation_service: ConversationService,
        repository: ConversationRepository,
        conversation: Conversation,
        user: User,
    ) -> None:
        await repository.add_user_message(conversation=conversation, content="q")

        detail = await conversation_service.get(conversation.id, user_id=user.id)

        assert detail.messages[0].answer_status is None
        assert detail.messages[0].trace == []

    async def test_missing_conversation_raises(
        self, conversation_service: ConversationService, user: User
    ) -> None:
        missing = uuid.uuid4()
        with pytest.raises(ConversationNotFoundError) as excinfo:
            await conversation_service.get(missing, user_id=user.id)
        assert excinfo.value.conversation_id == missing

    async def test_another_users_conversation_raises(
        self,
        conversation_service: ConversationService,
        conversation: Conversation,
        other_user: User,
    ) -> None:
        with pytest.raises(ConversationNotFoundError):
            await conversation_service.get(conversation.id, user_id=other_user.id)

    async def test_require_conversation_returns_the_orm_row(
        self,
        conversation_service: ConversationService,
        conversation: Conversation,
        user: User,
    ) -> None:
        row = await conversation_service.require_conversation(conversation.id, user_id=user.id)
        assert row.id == conversation.id

    async def test_require_conversation_enforces_ownership(
        self,
        conversation_service: ConversationService,
        conversation: Conversation,
        other_user: User,
    ) -> None:
        with pytest.raises(ConversationNotFoundError):
            await conversation_service.require_conversation(conversation.id, user_id=other_user.id)


class TestListForUser:
    async def _seed(self, repository: ConversationRepository, user: User, count: int) -> None:
        for index in range(count):
            await repository.create_conversation(user_id=user.id, title=f"Thread {index}")

    async def test_empty_result_reports_zero_pages(
        self, conversation_service: ConversationService, user: User
    ) -> None:
        page = await conversation_service.list_for_user(user_id=user.id)

        assert page.items == []
        assert page.total == 0
        assert page.total_pages == 0

    @pytest.mark.parametrize(
        ("total", "page_size", "expected_pages"),
        [(1, 20, 1), (20, 20, 1), (21, 20, 2), (5, 2, 3), (4, 2, 2)],
    )
    async def test_total_pages_is_a_ceiling_division(
        self,
        conversation_service: ConversationService,
        repository: ConversationRepository,
        user: User,
        total: int,
        page_size: int,
        expected_pages: int,
    ) -> None:
        await self._seed(repository, user, total)

        page = await conversation_service.list_for_user(
            user_id=user.id, page=1, page_size=page_size
        )

        assert page.total == total
        assert page.total_pages == expected_pages

    async def test_page_echoes_the_requested_window(
        self,
        conversation_service: ConversationService,
        repository: ConversationRepository,
        user: User,
    ) -> None:
        await self._seed(repository, user, 5)

        page = await conversation_service.list_for_user(user_id=user.id, page=3, page_size=2)

        assert (page.page, page.page_size) == (3, 2)
        assert len(page.items) == 1

    async def test_summaries_do_not_carry_messages(
        self,
        conversation_service: ConversationService,
        repository: ConversationRepository,
        user: User,
    ) -> None:
        await self._seed(repository, user, 1)
        page = await conversation_service.list_for_user(user_id=user.id)
        assert not hasattr(page.items[0], "messages")


class TestMappers:
    async def test_message_to_response_tolerates_a_null_trace(
        self, repository: ConversationRepository, conversation: Conversation
    ) -> None:
        message = await repository.add_user_message(conversation=conversation, content="q")
        message.trace = None  # type: ignore[assignment]

        response = message_to_response(message)

        assert response.trace == []

    async def test_conversation_to_detail_includes_counts(
        self, repository: ConversationRepository, conversation: Conversation
    ) -> None:
        await repository.add_user_message(conversation=conversation, content="q")
        await repository.refresh_messages(conversation)

        detail = conversation_to_detail(conversation)

        assert detail.message_count == 1
        assert len(detail.messages) == 1
