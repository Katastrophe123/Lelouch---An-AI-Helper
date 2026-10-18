"""HTTP client for the Homework Helper API.

The frontend's only dependency on the backend is this module. It holds no
business logic, no database driver and no LLM client - all of which v1's
Streamlit app did hold. Swapping the UI for anything else means reimplementing
only this file.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import httpx
from httpx_sse import connect_sse


class ApiError(Exception):
    """A non-2xx response, carrying the backend's structured error envelope."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(f"[{status_code}] {code}: {message}")
        self.status_code = status_code
        self.code = code
        self.message = message

    @classmethod
    def from_response(cls, response: httpx.Response) -> ApiError:
        code, message = "http_error", response.text[:500]
        try:
            payload = response.json()
            error = payload.get("error") or {}
            code = error.get("code", code)
            message = error.get("message", message)
        except (json.JSONDecodeError, AttributeError):
            pass
        return cls(response.status_code, code, message)


@dataclass(frozen=True, slots=True)
class Session:
    """An authenticated session."""

    access_token: str
    user_id: str
    username: str


@dataclass(frozen=True, slots=True)
class StreamEvent:
    """One decoded Server-Sent Event."""

    name: str
    data: dict[str, Any]


class HomeworkHelperClient:
    """Thin, synchronous wrapper over the API (Streamlit runs synchronously)."""

    def __init__(self, base_url: str, *, timeout: float = 120.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    # --- plumbing ---------------------------------------------------------
    def _client(self, token: str | None = None) -> httpx.Client:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        return httpx.Client(base_url=self._base_url, timeout=self._timeout, headers=headers)

    @staticmethod
    def _unwrap(response: httpx.Response) -> Any:
        if response.status_code >= 400:
            raise ApiError.from_response(response)
        return response.json()

    # --- endpoints --------------------------------------------------------
    def health(self) -> dict[str, Any]:
        with self._client() as client:
            return self._unwrap(client.get("/health"))  # type: ignore[no-any-return]

    def sign_in(self, username: str, password: str) -> Session:
        with self._client() as client:
            payload = self._unwrap(
                client.post("/auth/token", json={"username": username, "password": password})
            )
        return Session(
            access_token=payload["access_token"],
            user_id=payload["user_id"],
            username=payload["username"],
        )

    def create_conversation(
        self, token: str, *, title: str | None = None, subject: str | None = None
    ) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if title:
            body["title"] = title
        if subject:
            body["subject"] = subject
        with self._client(token) as client:
            return self._unwrap(client.post("/conversations", json=body))  # type: ignore[no-any-return]

    def list_conversations(
        self, token: str, *, page: int = 1, page_size: int = 20
    ) -> dict[str, Any]:
        with self._client(token) as client:
            return self._unwrap(  # type: ignore[no-any-return]
                client.get("/conversations", params={"page": page, "page_size": page_size})
            )

    def get_conversation(self, token: str, conversation_id: str) -> dict[str, Any]:
        with self._client(token) as client:
            return self._unwrap(client.get(f"/conversations/{conversation_id}"))  # type: ignore[no-any-return]

    def ask(self, token: str, conversation_id: str, content: str) -> dict[str, Any]:
        with self._client(token) as client:
            return self._unwrap(  # type: ignore[no-any-return]
                client.post(f"/conversations/{conversation_id}/messages", json={"content": content})
            )

    def ask_streaming(
        self, token: str, conversation_id: str, content: str
    ) -> Iterator[StreamEvent]:
        """Yield decoded events from the SSE endpoint as they arrive."""
        with (
            self._client(token) as client,
            connect_sse(
                client,
                "POST",
                f"/conversations/{conversation_id}/messages/stream",
                json={"content": content},
            ) as event_source,
        ):
            if event_source.response.status_code >= 400:
                event_source.response.read()
                raise ApiError.from_response(event_source.response)
            for event in event_source.iter_sse():
                if not event.data:
                    continue
                try:
                    data = json.loads(event.data)
                except json.JSONDecodeError:
                    continue
                yield StreamEvent(name=event.event or "message", data=data)
