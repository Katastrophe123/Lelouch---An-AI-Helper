#!/usr/bin/env python
"""Seed a demo learner with a few answered questions.

Useful for populating the UI after a fresh ``docker compose up``. Idempotent:
does nothing if the demo user already has conversations.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.db.repository import ConversationRepository, UserRepository  # noqa: E402
from app.db.session import Database  # noqa: E402
from app.llm.factory import build_llm_client  # noqa: E402
from app.logging_config import configure_logging  # noqa: E402
from app.models.enums import Subject  # noqa: E402
from app.services.tutor_service import TutorService  # noqa: E402

DEMO_USERNAME = "demo"

DEMO_CONVERSATIONS: list[tuple[str, Subject, list[str]]] = [
    (
        "Projectile motion revision",
        Subject.PHYSICS,
        [
            "A ball is dropped from rest. What is 2 * 9.81 and why does it matter here?",
            "How would the answer change if the ball were thrown downwards instead?",
        ],
    ),
    (
        "Arithmetic warm-up",
        Subject.MATH,
        ["What is 17 * 23?", "What is 15% of 240?"],
    ),
]


async def async_main() -> int:
    settings = get_settings()
    configure_logging(level="WARNING", json_output=False)

    database = Database.from_settings(settings)
    if not await database.healthcheck():
        print("Database is not reachable", file=sys.stderr)
        await database.dispose()
        return 2

    llm = build_llm_client(settings)
    service = TutorService(settings=settings, llm=llm)

    created_conversations = created_turns = 0
    try:
        async with database.session() as session:
            users = UserRepository(session)
            user = await users.get_or_create(DEMO_USERNAME, display_name="Demo learner")
            repository = ConversationRepository(session)
            _, total = await repository.list_conversations(user_id=user.id, page=1, page_size=1)
            if total:
                print(f"demo user already has {total} conversation(s); nothing to do")
                return 0

        for title, subject, questions in DEMO_CONVERSATIONS:
            # One transaction per conversation so a partial failure keeps the rest.
            async with database.session() as session:
                users = UserRepository(session)
                user = await users.get_or_create(DEMO_USERNAME)
                repository = ConversationRepository(session)
                conversation = await repository.create_conversation(
                    user_id=user.id, title=title, subject=subject
                )
                created_conversations += 1
                for question in questions:
                    await service.run_turn(
                        conversation=conversation, content=question, repository=repository
                    )
                    created_turns += 1
    finally:
        await llm.aclose()
        await database.dispose()

    print(
        f"seeded user '{DEMO_USERNAME}' with {created_conversations} conversation(s) "
        f"and {created_turns} answered question(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(async_main()))
