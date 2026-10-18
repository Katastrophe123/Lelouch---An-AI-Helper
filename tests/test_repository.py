"""Repository tests against a real (in-memory) database.

These cover the capability v1 did not have at all: durable, queryable
conversation history.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.exc import IntegrityError

from app.db.models import Conversation, User
from app.db.repository import ConversationRepository, UserRepository
from app.models.domain import StepRecord
from app.models.enums import (
    AnswerStatus,
    ConversationStatus,
    FallbackReason,
    MessageRole,
    NodeName,
    RoutingDecision,
    StepStatus,
    Subject,
)


def make_trace() -> list[StepRecord]:
    return [
        StepRecord(
            sequence=1,
            node=NodeName.UNDERSTAND,
            status=StepStatus.OK,
            latency_ms=11,
            output={"subject": "math"},
            routing_decision=RoutingDecision.TO_SOLVE,
        ),
        StepRecord(
            sequence=2,
            node=NodeName.FALLBACK,
            status=StepStatus.OK,
            latency_ms=2,
            output={"reason": "self_check_failed"},
            routing_decision=RoutingDecision.END,
        ),
    ]


class TestUserRepository:
    async def test_get_or_create_inserts_on_first_sign_in(self, users: UserRepository) -> None:
        user = await users.get_or_create("ada", display_name="Ada L")

        assert isinstance(user.id, uuid.UUID)
        assert user.username == "ada"
        assert user.display_name == "Ada L"

    async def test_get_or_create_is_idempotent(self, users: UserRepository) -> None:
        first = await users.get_or_create("ada")
        second = await users.get_or_create("ada")
        assert first.id == second.id

    async def test_display_name_defaults_to_the_username(self, users: UserRepository) -> None:
        user = await users.get_or_create("ada")
        assert user.display_name == "ada"

    async def test_lookup_by_username_and_id(self, users: UserRepository) -> None:
        user = await users.get_or_create("ada")
        assert (await users.get_by_username("ada")).id == user.id  # type: ignore[union-attr]
        assert (await users.get_by_id(user.id)).username == "ada"  # type: ignore[union-attr]

    async def test_unknown_lookups_return_none(self, users: UserRepository) -> None:
        assert await users.get_by_username("nobody") is None
        assert await users.get_by_id(uuid.uuid4()) is None

    async def test_usernames_are_unique(self, users: UserRepository, session: object) -> None:
        await users.get_or_create("ada")
        session.add(User(username="ada"))  # type: ignore[attr-defined]
        with pytest.raises(IntegrityError):
            await session.flush()  # type: ignore[attr-defined]
        # The failed flush poisons the transaction; roll back so the fixture
        # can still close the session cleanly.
        await session.rollback()  # type: ignore[attr-defined]


class TestConversations:
    async def test_create_defaults(self, repository: ConversationRepository, user: User) -> None:
        conversation = await repository.create_conversation(user_id=user.id)

        assert conversation.title == "New conversation"
        assert conversation.status == ConversationStatus.ACTIVE.value
        assert conversation.message_count == 0
        assert conversation.subject is None

    async def test_create_with_title_and_subject(
        self, repository: ConversationRepository, user: User
    ) -> None:
        conversation = await repository.create_conversation(
            user_id=user.id, title="Calculus", subject=Subject.MATH
        )
        assert conversation.title == "Calculus"
        assert conversation.subject == Subject.MATH.value

    async def test_long_titles_are_truncated(
        self, repository: ConversationRepository, user: User
    ) -> None:
        conversation = await repository.create_conversation(user_id=user.id, title="x" * 400)
        assert len(conversation.title) == 200

    async def test_get_scoped_to_owner(
        self, repository: ConversationRepository, user: User, other_user: User
    ) -> None:
        conversation = await repository.create_conversation(user_id=user.id)

        assert await repository.get_conversation(conversation.id, user_id=user.id) is not None
        assert await repository.get_conversation(conversation.id, user_id=other_user.id) is None

    async def test_get_unknown_returns_none(self, repository: ConversationRepository) -> None:
        assert await repository.get_conversation(uuid.uuid4()) is None

    async def test_touch_backfills_the_subject_only_once(
        self, repository: ConversationRepository, user: User
    ) -> None:
        conversation = await repository.create_conversation(user_id=user.id)
        await repository.touch_conversation(conversation, subject=Subject.PHYSICS)
        assert conversation.subject == Subject.PHYSICS.value

        # A later turn in a different subject must not relabel the thread.
        await repository.touch_conversation(conversation, subject=Subject.MATH)
        assert conversation.subject == Subject.PHYSICS.value

    async def test_touch_advances_updated_at(
        self, repository: ConversationRepository, user: User
    ) -> None:
        conversation = await repository.create_conversation(user_id=user.id)
        before = conversation.updated_at
        await repository.touch_conversation(conversation)
        assert conversation.updated_at >= before


class TestMessages:
    async def test_sequences_increment_per_conversation(
        self, repository: ConversationRepository, conversation: Conversation
    ) -> None:
        first = await repository.add_user_message(conversation=conversation, content="q1")
        second = await repository.add_assistant_message(
            conversation=conversation, content="a1", answer_status=AnswerStatus.VERIFIED
        )
        third = await repository.add_user_message(conversation=conversation, content="q2")

        assert [first.sequence, second.sequence, third.sequence] == [1, 2, 3]
        assert conversation.message_count == 3

    async def test_sequences_are_independent_across_conversations(
        self, repository: ConversationRepository, user: User
    ) -> None:
        first = await repository.create_conversation(user_id=user.id)
        second = await repository.create_conversation(user_id=user.id)

        a = await repository.add_user_message(conversation=first, content="q")
        b = await repository.add_user_message(conversation=second, content="q")

        assert a.sequence == b.sequence == 1

    async def test_assistant_metadata_and_trace_are_stored(
        self, repository: ConversationRepository, conversation: Conversation
    ) -> None:
        message = await repository.add_assistant_message(
            conversation=conversation,
            content="Here is the answer.",
            answer_status=AnswerStatus.UNVERIFIED,
            fallback_reason=FallbackReason.SELF_CHECK_FAILED,
            confidence=0.31,
            retry_count=2,
            latency_ms=1234,
            error="marker never satisfied",
            trace=make_trace(),
        )

        assert message.role == MessageRole.ASSISTANT.value
        assert message.answer_status == AnswerStatus.UNVERIFIED.value
        assert message.fallback_reason == FallbackReason.SELF_CHECK_FAILED.value
        assert message.confidence == pytest.approx(0.31)
        assert message.retry_count == 2
        assert message.latency_ms == 1234
        assert message.error == "marker never satisfied"
        assert message.verified is False
        assert [step["node"] for step in message.trace] == ["understand", "fallback"]
        assert message.trace[0]["routing_decision"] == "to_solve"

    async def test_verified_flag_tracks_the_status(
        self, repository: ConversationRepository, conversation: Conversation
    ) -> None:
        verified = await repository.add_assistant_message(
            conversation=conversation, content="a", answer_status=AnswerStatus.VERIFIED
        )
        assert verified.verified is True
        assert verified.is_verified is True
        assert verified.is_assistant is True

    async def test_user_messages_have_no_run_metadata(
        self, repository: ConversationRepository, conversation: Conversation
    ) -> None:
        message = await repository.add_user_message(conversation=conversation, content="q")
        assert message.answer_status is None
        assert message.trace == []
        assert message.is_assistant is False

    async def test_duplicate_sequence_is_rejected(
        self, repository: ConversationRepository, conversation: Conversation, session: object
    ) -> None:
        """The unique constraint is what stops a trace being written twice."""
        from app.db.models import Message

        await repository.add_user_message(conversation=conversation, content="q")
        session.add(  # type: ignore[attr-defined]
            Message(
                conversation_id=conversation.id,
                sequence=1,
                role=MessageRole.USER.value,
                content="duplicate",
                trace=[],
            )
        )
        with pytest.raises(IntegrityError):
            await session.flush()  # type: ignore[attr-defined]
        await session.rollback()  # type: ignore[attr-defined]

    async def test_messages_are_returned_in_order(
        self, repository: ConversationRepository, conversation: Conversation
    ) -> None:
        for index in range(5):
            await repository.add_user_message(conversation=conversation, content=f"q{index}")
        await repository.refresh_messages(conversation)

        assert [m.sequence for m in conversation.messages] == [1, 2, 3, 4, 5]


class TestHistory:
    async def test_history_is_oldest_first(
        self, repository: ConversationRepository, conversation: Conversation
    ) -> None:
        await repository.add_user_message(conversation=conversation, content="q1")
        await repository.add_assistant_message(
            conversation=conversation, content="a1", answer_status=AnswerStatus.VERIFIED
        )
        await repository.add_user_message(conversation=conversation, content="q2")

        history = await repository.recent_history(conversation.id, limit=10)

        assert [turn.content for turn in history] == ["q1", "a1", "q2"]
        assert history[1].role is MessageRole.ASSISTANT

    async def test_history_window_keeps_the_most_recent(
        self, repository: ConversationRepository, conversation: Conversation
    ) -> None:
        for index in range(10):
            await repository.add_user_message(conversation=conversation, content=f"q{index}")

        history = await repository.recent_history(conversation.id, limit=3)

        assert [turn.content for turn in history] == ["q7", "q8", "q9"]

    async def test_zero_window_returns_nothing(
        self, repository: ConversationRepository, conversation: Conversation
    ) -> None:
        await repository.add_user_message(conversation=conversation, content="q")
        assert await repository.recent_history(conversation.id, limit=0) == []

    async def test_empty_conversation_has_no_history(
        self, repository: ConversationRepository, conversation: Conversation
    ) -> None:
        assert await repository.recent_history(conversation.id, limit=5) == []


class TestListConversations:
    async def _seed(self, repository: ConversationRepository, user: User, count: int) -> None:
        for index in range(count):
            await repository.create_conversation(
                user_id=user.id,
                title=f"Thread {index}",
                subject=Subject.MATH if index % 2 == 0 else Subject.PHYSICS,
            )

    async def test_scoped_to_the_owner(
        self, repository: ConversationRepository, user: User, other_user: User
    ) -> None:
        await self._seed(repository, user, 3)
        await self._seed(repository, other_user, 2)

        _, mine = await repository.list_conversations(user_id=user.id)
        _, theirs = await repository.list_conversations(user_id=other_user.id)

        assert mine == 3
        assert theirs == 2

    async def test_filters_by_subject(self, repository: ConversationRepository, user: User) -> None:
        await self._seed(repository, user, 4)
        _, total = await repository.list_conversations(user_id=user.id, subject=Subject.MATH)
        assert total == 2

    async def test_filters_by_status(self, repository: ConversationRepository, user: User) -> None:
        await self._seed(repository, user, 2)
        _, active = await repository.list_conversations(
            user_id=user.id, status=ConversationStatus.ACTIVE
        )
        _, archived = await repository.list_conversations(
            user_id=user.id, status=ConversationStatus.ARCHIVED
        )
        assert (active, archived) == (2, 0)

    async def test_pagination_splits_the_result_set(
        self, repository: ConversationRepository, user: User
    ) -> None:
        await self._seed(repository, user, 5)

        first, total = await repository.list_conversations(user_id=user.id, page=1, page_size=2)
        second, _ = await repository.list_conversations(user_id=user.id, page=2, page_size=2)
        third, _ = await repository.list_conversations(user_id=user.id, page=3, page_size=2)

        assert total == 5
        assert (len(first), len(second), len(third)) == (2, 2, 1)
        ids = {c.id for c in first} | {c.id for c in second} | {c.id for c in third}
        assert len(ids) == 5

    async def test_page_beyond_the_end_is_empty(
        self, repository: ConversationRepository, user: User
    ) -> None:
        await self._seed(repository, user, 3)
        rows, total = await repository.list_conversations(user_id=user.id, page=99)
        assert rows == []
        assert total == 3

    async def test_ordered_by_most_recently_updated(
        self, repository: ConversationRepository, user: User
    ) -> None:
        await self._seed(repository, user, 3)
        rows, _ = await repository.list_conversations(user_id=user.id)
        timestamps = [c.updated_at for c in rows]
        assert timestamps == sorted(timestamps, reverse=True)


class TestAggregates:
    async def test_count_messages_by_status(
        self, repository: ConversationRepository, conversation: Conversation, user: User
    ) -> None:
        await repository.add_user_message(conversation=conversation, content="q")
        await repository.add_assistant_message(
            conversation=conversation, content="a", answer_status=AnswerStatus.VERIFIED
        )
        await repository.add_assistant_message(
            conversation=conversation, content="b", answer_status=AnswerStatus.UNVERIFIED
        )
        await repository.add_assistant_message(
            conversation=conversation, content="c", answer_status=AnswerStatus.VERIFIED
        )

        counts = await repository.count_messages_by_status(user.id)

        # User messages have no answer_status and are excluded.
        assert counts == {"verified": 2, "unverified": 1}

    async def test_counts_are_scoped_to_the_user(
        self, repository: ConversationRepository, conversation: Conversation, other_user: User
    ) -> None:
        await repository.add_assistant_message(
            conversation=conversation, content="a", answer_status=AnswerStatus.VERIFIED
        )
        assert await repository.count_messages_by_status(other_user.id) == {}
