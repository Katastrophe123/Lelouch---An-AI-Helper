"""Knowledge base: parsing, chunking, embeddings, storage and scoped search."""

from __future__ import annotations

import io
import uuid
from typing import Any

import pytest

from app.config import Settings
from app.db.models import User
from app.db.repository import DocumentRepository, cosine_similarity
from app.db.session import Database
from app.knowledge.chunking import chunk_text
from app.knowledge.embeddings import (
    EmbedderUnavailableError,
    FastEmbedEmbedder,
    HashingEmbedder,
    build_embedder,
)
from app.knowledge.parsing import EmptyDocumentError, UnsupportedDocumentError, extract_text
from app.knowledge.service import DocumentNotFoundError, DocumentTooLargeError, KnowledgeService

NOTES = (
    b"# Biology\n\nMitochondria are the powerhouse of the cell; they produce ATP.\n\n"
    b"# History\n\nThe Battle of Plassey was fought in 1757 in Bengal."
)


class TestChunking:
    def test_short_text_is_one_chunk(self) -> None:
        assert chunk_text("Just one line.", size=100, overlap=10) == ["Just one line."]

    def test_chunks_respect_the_size_limit(self) -> None:
        text = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(12))
        chunks = chunk_text(text, size=500, overlap=80)
        assert len(chunks) > 1
        assert all(len(chunk) <= 500 for chunk in chunks)

    def test_consecutive_chunks_overlap(self) -> None:
        text = "\n\n".join(f"Paragraph {i} " + "alpha beta gamma " * 10 for i in range(6))
        chunks = chunk_text(text, size=300, overlap=60)
        for previous, current in zip(chunks, chunks[1:], strict=False):
            tail_words = previous.split()[-3:]
            assert " ".join(tail_words) in current

    def test_a_giant_paragraph_is_split_on_whitespace(self) -> None:
        chunks = chunk_text("token " * 1000, size=200, overlap=0)
        assert all(len(chunk) <= 200 for chunk in chunks)
        assert all(not chunk.endswith("toke") for chunk in chunks)

    def test_no_text_means_no_chunks(self) -> None:
        assert chunk_text("   \n\n  ", size=100) == []

    def test_invalid_size_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            chunk_text("x", size=0)


class TestParsing:
    def test_plain_text_and_markdown(self) -> None:
        assert "Plassey" in extract_text("notes.md", NOTES)

    def test_utf8_bom_is_stripped(self) -> None:
        assert extract_text("a.txt", "﻿héllo".encode()) == "héllo"

    def test_docx_paragraphs_and_tables(self) -> None:
        from docx import Document

        document = Document()
        document.add_paragraph("Quarterly revenue grew 12%.")
        table = document.add_table(rows=1, cols=2)
        table.rows[0].cells[0].text = "Region"
        table.rows[0].cells[1].text = "North"
        buffer = io.BytesIO()
        document.save(buffer)

        text = extract_text("report.docx", buffer.getvalue())
        assert "Quarterly revenue grew 12%." in text
        assert "Region | North" in text

    def test_unsupported_extension(self) -> None:
        with pytest.raises(UnsupportedDocumentError, match="Unsupported file type"):
            extract_text("photo.png", b"\x89PNG")

    def test_corrupt_pdf_is_reported_as_unreadable(self) -> None:
        with pytest.raises(UnsupportedDocumentError, match="Could not read"):
            extract_text("broken.pdf", b"not a pdf")

    def test_blank_file_is_empty(self) -> None:
        with pytest.raises(EmptyDocumentError):
            extract_text("blank.txt", b"  \n ")


class TestEmbeddings:
    async def test_hashing_vectors_are_normalised_and_deterministic(self) -> None:
        embedder = HashingEmbedder()
        first, second = await embedder.embed_documents(["the cell makes ATP", "the cell makes ATP"])
        assert first == second
        assert len(first) == 384
        assert abs(sum(v * v for v in first) - 1.0) < 1e-9

    async def test_related_text_scores_higher(self) -> None:
        embedder = HashingEmbedder()
        query = await embedder.embed_query("battle of plassey")
        history, biology = await embedder.embed_documents(
            ["The Battle of Plassey was in 1757.", "Mitochondria produce ATP."]
        )
        assert cosine_similarity(query, history) > cosine_similarity(query, biology)

    async def test_empty_text_gives_a_zero_vector(self) -> None:
        (vector,) = await HashingEmbedder().embed_documents([""])
        assert not any(vector)
        assert cosine_similarity(vector, vector) == 0.0

    def test_factory(self, settings: Settings) -> None:
        assert isinstance(build_embedder(settings), HashingEmbedder)
        fast = build_embedder(settings.model_copy(update={"embedding_backend": "fastembed"}))
        assert isinstance(fast, FastEmbedEmbedder)

    async def test_fastembed_load_failure_is_a_clear_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import fastembed

        def broken(**_: Any) -> None:
            raise ValueError("Could not load model from any source.")

        monkeypatch.setattr(fastembed, "TextEmbedding", broken)
        with pytest.raises(EmbedderUnavailableError, match="downloaded once on first use"):
            await FastEmbedEmbedder("BAAI/bge-small-en-v1.5").embed_query("hi")

    async def test_fastembed_dimension_mismatch_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import fastembed

        class WideModel:
            def __init__(self, **_: Any) -> None: ...

            def embed(self, texts: list[str]) -> Any:
                return iter([[0.1] * 768 for _ in texts])

        monkeypatch.setattr(fastembed, "TextEmbedding", WideModel)
        with pytest.raises(EmbedderUnavailableError, match="768-d"):
            await FastEmbedEmbedder("some/768-model").embed_documents(["x"])

    async def test_fastembed_uses_query_embedding_for_queries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import fastembed

        class Model:
            def __init__(self, **_: Any) -> None: ...

            def embed(self, texts: list[str]) -> Any:
                return iter([[0.0] * 384 for _ in texts])

            def query_embed(self, texts: list[str]) -> Any:
                return iter([[1.0] * 384 for _ in texts])

        monkeypatch.setattr(fastembed, "TextEmbedding", Model)
        embedder = FastEmbedEmbedder("m")
        assert (await embedder.embed_query("q"))[0] == 1.0
        assert (await embedder.embed_documents(["d"]))[0][0] == 0.0


class TestKnowledgeService:
    async def test_ingest_then_search_finds_the_right_file(
        self, knowledge_service: KnowledgeService, user: User, session: Any
    ) -> None:
        await session.commit()
        await knowledge_service.ingest(
            user_id=user.id, filename="notes.md", content_type="text/markdown", data=NOTES
        )
        await knowledge_service.ingest(
            user_id=user.id,
            filename="chai.txt",
            content_type="text/plain",
            data=b"Masala chai: tea leaves, ginger, cardamom, milk and sugar.",
        )
        hits = await knowledge_service.search(user_id=user.id, query="masala chai ginger")
        assert hits[0].filename == "chai.txt"
        assert hits[0].score > hits[-1].score

    async def test_search_is_scoped_to_the_owner(
        self, knowledge_service: KnowledgeService, user: User, other_user: User, session: Any
    ) -> None:
        await session.commit()
        await knowledge_service.ingest(
            user_id=user.id, filename="secret.md", content_type=None, data=NOTES
        )
        assert await knowledge_service.search(user_id=other_user.id, query="Plassey") == []

    async def test_long_documents_are_chunked(
        self, knowledge_service: KnowledgeService, user: User, session: Any
    ) -> None:
        await session.commit()
        data = "\n\n".join(f"Section {i}: " + "detail " * 30 for i in range(20)).encode()
        document = await knowledge_service.ingest(
            user_id=user.id, filename="long.txt", content_type=None, data=data
        )
        assert document.chunk_count > 5
        listed = await knowledge_service.list_documents(user_id=user.id)
        assert [doc.id for doc in listed] == [document.id]

    async def test_oversized_upload_is_rejected(
        self, knowledge_service: KnowledgeService, user: User
    ) -> None:
        with pytest.raises(DocumentTooLargeError, match="limit is 1 MB"):
            await knowledge_service.ingest(
                user_id=user.id, filename="big.txt", content_type=None, data=b"x" * 1_100_000
            )

    async def test_delete_removes_document_and_chunks(
        self,
        knowledge_service: KnowledgeService,
        database: Database,
        user: User,
        session: Any,
    ) -> None:
        await session.commit()
        document = await knowledge_service.ingest(
            user_id=user.id, filename="notes.md", content_type=None, data=NOTES
        )
        await knowledge_service.delete(document.id, user_id=user.id)
        assert await knowledge_service.list_documents(user_id=user.id) == []
        assert await knowledge_service.search(user_id=user.id, query="Plassey") == []

    async def test_deleting_someone_elses_document_is_not_found(
        self, knowledge_service: KnowledgeService, user: User, other_user: User, session: Any
    ) -> None:
        await session.commit()
        document = await knowledge_service.ingest(
            user_id=user.id, filename="notes.md", content_type=None, data=NOTES
        )
        with pytest.raises(DocumentNotFoundError):
            await knowledge_service.delete(document.id, user_id=other_user.id)
        with pytest.raises(DocumentNotFoundError):
            await knowledge_service.delete(uuid.uuid4(), user_id=user.id)


class TestDocumentRepository:
    async def test_mismatched_embeddings_are_rejected(self, session: Any, user: User) -> None:
        with pytest.raises(ValueError, match="same length"):
            await DocumentRepository(session).add_document(
                user_id=user.id,
                filename="a.txt",
                content_type=None,
                size_bytes=1,
                chunks=["a", "b"],
                embeddings=[[0.0] * 384],
            )

    async def test_top_k_limits_results(self, session: Any, user: User) -> None:
        repo = DocumentRepository(session)
        embedder = HashingEmbedder()
        chunks = [f"fact number {i}" for i in range(10)]
        await repo.add_document(
            user_id=user.id,
            filename="facts.txt",
            content_type=None,
            size_bytes=100,
            chunks=chunks,
            embeddings=await embedder.embed_documents(chunks),
        )
        hits = await repo.search(
            user_id=user.id, embedding=await embedder.embed_query("fact number 3"), top_k=3
        )
        assert len(hits) == 3
        assert hits[0].content == "fact number 3"
