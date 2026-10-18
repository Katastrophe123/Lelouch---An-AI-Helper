"""Conversation and message endpoints, including the SSE streaming turn."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Path, Query, Request, status
from sse_starlette.sse import EventSourceResponse

from app.api.deps import (
    ConversationRepositoryDep,
    ConversationServiceDep,
    CurrentUserDep,
    DatabaseDep,
    PaginationDep,
    TutorServiceDep,
)
from app.api.errors import NotFoundError
from app.logging_config import get_logger
from app.models.enums import ConversationStatus, Subject
from app.models.schemas import (
    ConversationCreate,
    ConversationDetail,
    ConversationSummary,
    MessageCreate,
    Page,
    TurnResponse,
)
from app.services.conversation_service import message_to_response
from app.services.tutor_service import ConversationNotFoundError

logger = get_logger(__name__)

router = APIRouter(prefix="/conversations", tags=["conversations"])


def _not_found(conversation_id: uuid.UUID) -> NotFoundError:
    return NotFoundError(
        f"Conversation {conversation_id} was not found",
        details={"conversation_id": str(conversation_id)},
    )


@router.post(
    "",
    response_model=ConversationDetail,
    status_code=status.HTTP_201_CREATED,
    summary="Start a conversation",
    responses={401: {"description": "Missing or invalid token"}},
)
async def create_conversation(
    payload: ConversationCreate,
    user: CurrentUserDep,
    service: ConversationServiceDep,
) -> ConversationDetail:
    """Create an empty conversation owned by the authenticated learner."""
    conversation = await service.create(
        user_id=user.id, title=payload.title, subject=payload.subject
    )
    logger.info("conversation_created", conversation_id=str(conversation.id))
    return conversation


@router.get(
    "",
    response_model=Page[ConversationSummary],
    summary="List the current learner's conversations",
    responses={401: {"description": "Missing or invalid token"}},
)
async def list_conversations(
    user: CurrentUserDep,
    service: ConversationServiceDep,
    pagination: PaginationDep,
    status_filter: Annotated[
        ConversationStatus | None, Query(alias="status", description="Filter by status")
    ] = None,
    subject: Annotated[Subject | None, Query(description="Filter by subject")] = None,
) -> Page[ConversationSummary]:
    """Paginated, most-recently-updated first, scoped to the caller."""
    return await service.list_for_user(
        user_id=user.id,
        status=status_filter,
        subject=subject,
        page=pagination.page,
        page_size=pagination.page_size,
    )


@router.get(
    "/{conversation_id}",
    response_model=ConversationDetail,
    summary="Fetch a conversation with its full message history",
    responses={
        401: {"description": "Missing or invalid token"},
        404: {"description": "Conversation not found or not owned by the caller"},
    },
)
async def get_conversation(
    user: CurrentUserDep,
    service: ConversationServiceDep,
    conversation_id: Annotated[uuid.UUID, Path(description="Conversation identifier")],
) -> ConversationDetail:
    try:
        return await service.get(conversation_id, user_id=user.id)
    except ConversationNotFoundError as exc:
        raise _not_found(conversation_id) from exc


@router.post(
    "/{conversation_id}/messages",
    response_model=TurnResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ask a question (runs the tutoring graph)",
    responses={
        401: {"description": "Missing or invalid token"},
        404: {"description": "Conversation not found or not owned by the caller"},
        422: {"description": "Invalid message payload"},
    },
)
async def create_message(
    payload: MessageCreate,
    user: CurrentUserDep,
    service: ConversationServiceDep,
    tutor: TutorServiceDep,
    repository: ConversationRepositoryDep,
    conversation_id: Annotated[uuid.UUID, Path(description="Conversation identifier")],
) -> TurnResponse:
    """Run understand -> solve -> self-check -> explain and store both messages.

    Synchronous by design so the response carries the answer and its trace; the
    streaming variant below is the better fit for an interactive UI.
    """
    try:
        conversation = await service.require_conversation(conversation_id, user_id=user.id)
    except ConversationNotFoundError as exc:
        raise _not_found(conversation_id) from exc

    result = await tutor.run_turn(
        conversation=conversation, content=payload.content, repository=repository
    )
    return TurnResponse(
        user_message=message_to_response(result.user_message),
        assistant_message=message_to_response(result.assistant_message),
    )


@router.post(
    "/{conversation_id}/messages/stream",
    summary="Ask a question and stream the explanation (SSE)",
    response_class=EventSourceResponse,
    responses={
        200: {
            "description": ("An `text/event-stream` of `step`, `token`, `done` and `error` events"),
            "content": {"text/event-stream": {}},
        },
        401: {"description": "Missing or invalid token"},
        404: {"description": "Conversation not found or not owned by the caller"},
    },
)
async def stream_message(
    request: Request,
    payload: MessageCreate,
    user: CurrentUserDep,
    service: ConversationServiceDep,
    tutor: TutorServiceDep,
    database: DatabaseDep,
    conversation_id: Annotated[uuid.UUID, Path(description="Conversation identifier")],
) -> EventSourceResponse:
    """Stream the turn as Server-Sent Events.

    Ownership is checked *before* the stream opens, so a 404 is a normal JSON
    error response rather than an error event inside a 200 stream.

    Event sequence: one ``step`` per graph node, then ``token`` events carrying
    the explanation as it is generated, then a single ``done`` event with the
    persisted message ids. Failures arrive as one ``error`` event.
    """
    try:
        await service.require_conversation(conversation_id, user_id=user.id)
    except ConversationNotFoundError as exc:
        raise _not_found(conversation_id) from exc

    async def event_stream() -> AsyncIterator[dict[str, str]]:
        async for event in tutor.stream_turn(
            database=database,
            conversation_id=conversation_id,
            user_id=user.id,
            content=payload.content,
        ):
            if await request.is_disconnected():  # pragma: no cover - client hang-up
                logger.info("stream_client_disconnected", conversation_id=str(conversation_id))
                break
            name = event.pop("event")
            yield {"event": name, "data": json.dumps(event)}

    return EventSourceResponse(event_stream())
