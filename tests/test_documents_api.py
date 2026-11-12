"""HTTP tests for /documents: upload, list, search, delete, errors, isolation."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from httpx import AsyncClient

from app.knowledge.embeddings import EmbedderUnavailableError, HashingEmbedder

NOTES = b"The Battle of Plassey was fought in 1757 in Bengal.\n\nMitochondria produce ATP."


async def upload(
    client: AsyncClient, headers: dict[str, str], name: str = "notes.md", data: bytes = NOTES
) -> Any:
    return await client.post(
        "/documents", files={"file": (name, data, "text/markdown")}, headers=headers
    )


class TestDocumentsApi:
    async def test_requires_auth(self, client: AsyncClient) -> None:
        assert (await client.get("/documents")).status_code == 401
        assert (await upload(client, {})).status_code == 401

    async def test_upload_list_search_delete(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        created = await upload(client, auth_headers)
        assert created.status_code == 201, created.text
        body = created.json()
        assert body["filename"] == "notes.md"
        assert body["chunk_count"] >= 1
        assert body["size_bytes"] == len(NOTES)

        listed = await client.get("/documents", headers=auth_headers)
        assert [doc["id"] for doc in listed.json()] == [body["id"]]

        hits = await client.post(
            "/documents/search", json={"query": "Plassey 1757"}, headers=auth_headers
        )
        assert hits.status_code == 200
        assert hits.json()[0]["filename"] == "notes.md"
        assert "Plassey" in hits.json()[0]["content"]

        deleted = await client.delete(f"/documents/{body['id']}", headers=auth_headers)
        assert deleted.status_code == 204
        assert (await client.get("/documents", headers=auth_headers)).json() == []

    async def test_unsupported_type_is_415(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        response = await upload(client, auth_headers, name="photo.png", data=b"\x89PNG")
        assert response.status_code == 415
        assert response.json()["error"]["code"] == "unsupported_media_type"

    async def test_empty_file_is_400(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        response = await upload(client, auth_headers, name="blank.txt", data=b"   ")
        assert response.status_code == 400

    async def test_oversized_file_is_413(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        response = await upload(client, auth_headers, name="big.txt", data=b"a " * 600_000)
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "payload_too_large"

    async def test_embedder_failure_is_503(
        self, app: FastAPI, client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        class Broken(HashingEmbedder):
            async def embed_documents(self, texts: list[str]) -> list[list[float]]:
                raise EmbedderUnavailableError("model download failed")

        app.state.knowledge_service.embedder = Broken()
        response = await upload(client, auth_headers)
        assert response.status_code == 503
        assert "model download failed" in response.json()["error"]["message"]
        search = await client.post("/documents/search", json={"query": "x"}, headers=auth_headers)
        assert search.status_code == 503

    async def test_other_users_cannot_see_or_delete(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
        other_auth_headers: dict[str, str],
    ) -> None:
        document = (await upload(client, auth_headers)).json()

        assert (await client.get("/documents", headers=other_auth_headers)).json() == []
        hits = await client.post(
            "/documents/search", json={"query": "Plassey"}, headers=other_auth_headers
        )
        assert hits.json() == []
        response = await client.delete(f"/documents/{document['id']}", headers=other_auth_headers)
        assert response.status_code == 404
        assert len((await client.get("/documents", headers=auth_headers)).json()) == 1
