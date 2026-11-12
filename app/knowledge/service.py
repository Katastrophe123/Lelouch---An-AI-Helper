"""Ingest, list, delete and search the user's documents.

The service owns its database sessions (rather than borrowing the request's)
because the agent calls :meth:`search` from inside a tool while a turn is
running, and a SQLAlchemy ``AsyncSession`` must not be shared concurrently.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from app.config import Settings
from app.db.models import Document
from app.db.repository import ChunkHit, DocumentRepository
from app.db.session import Database
from app.knowledge.chunking import chunk_text
from app.knowledge.embeddings import Embedder
from app.knowledge.parsing import extract_text
from app.logging_config import get_logger

logger = get_logger(__name__)

#: Embed in batches so one large upload does not hold a giant array in memory.
EMBED_BATCH = 64


class DocumentTooLargeError(Exception):
    """The upload exceeds ``MAX_UPLOAD_MB``."""


class DocumentNotFoundError(Exception):
    def __init__(self, document_id: uuid.UUID) -> None:
        super().__init__(f"Document {document_id} was not found")
        self.document_id = document_id


class KnowledgeService:
    def __init__(self, *, settings: Settings, database: Database, embedder: Embedder) -> None:
        self._settings = settings
        self._database = database
        self.embedder = embedder

    async def ingest(
        self, *, user_id: uuid.UUID, filename: str, content_type: str | None, data: bytes
    ) -> Document:
        """Parse, chunk, embed and store one file."""
        if len(data) > self._settings.max_upload_bytes:
            raise DocumentTooLargeError(
                f"{filename} is {len(data) / 1_048_576:.1f} MB; "
                f"the limit is {self._settings.max_upload_mb:g} MB"
            )
        text = extract_text(filename, data)
        chunks = chunk_text(
            text, size=self._settings.rag_chunk_chars, overlap=self._settings.rag_chunk_overlap
        )
        embeddings: list[list[float]] = []
        for start in range(0, len(chunks), EMBED_BATCH):
            embeddings.extend(
                await self.embedder.embed_documents(chunks[start : start + EMBED_BATCH])
            )
        async with self._database.session() as session:
            document = await DocumentRepository(session).add_document(
                user_id=user_id,
                filename=filename,
                content_type=content_type,
                size_bytes=len(data),
                chunks=chunks,
                embeddings=embeddings,
            )
        logger.info(
            "document_ingested",
            document_id=str(document.id),
            filename=filename,
            chunks=len(chunks),
            embedder=self.embedder.name,
        )
        return document

    async def list_documents(self, *, user_id: uuid.UUID) -> Sequence[Document]:
        async with self._database.session() as session:
            return await DocumentRepository(session).list_documents(user_id=user_id)

    async def delete(self, document_id: uuid.UUID, *, user_id: uuid.UUID) -> None:
        async with self._database.session() as session:
            deleted = await DocumentRepository(session).delete_document(
                document_id, user_id=user_id
            )
        if not deleted:
            raise DocumentNotFoundError(document_id)

    async def search(
        self, *, user_id: uuid.UUID, query: str, top_k: int | None = None
    ) -> list[ChunkHit]:
        vector = await self.embedder.embed_query(query)
        async with self._database.session() as session:
            return await DocumentRepository(session).search(
                user_id=user_id, embedding=vector, top_k=top_k or self._settings.rag_top_k
            )
