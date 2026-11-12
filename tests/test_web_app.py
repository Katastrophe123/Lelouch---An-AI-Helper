"""The built web app is served at / without shadowing any API route."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.config import Settings
from app.main import create_app, mount_web_app


def build_dist(tmp_path: Path) -> Path:
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>Lelouch</title>")
    (dist / "assets" / "app.js").write_text("console.log('hi')")
    return dist


async def test_serves_index_and_assets(settings: Settings, tmp_path: Path) -> None:
    app = create_app(settings.model_copy(update={"web_dist_dir": str(build_dist(tmp_path))}))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
        index = await client.get("/")
        assert index.status_code == 200
        assert "Lelouch" in index.text
        asset = await client.get("/assets/app.js")
        assert asset.status_code == 200


async def test_api_routes_win_over_the_web_app(
    app: FastAPI, tmp_path: Path, auth_headers: dict[str, str]
) -> None:
    assert mount_web_app(app, str(build_dist(tmp_path)))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
        unauthorised = await client.get("/conversations")
        assert unauthorised.status_code == 401
        assert unauthorised.json()["error"]["code"] == "unauthorized"
        health = await client.get("/health")
        assert health.json()["status"] == "ok"
        assert (await client.get("/")).status_code == 200


async def test_missing_build_is_skipped(settings: Settings, tmp_path: Path) -> None:
    app = create_app(settings.model_copy(update={"web_dist_dir": str(tmp_path / "nope")}))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
        assert (await client.get("/")).status_code == 404
