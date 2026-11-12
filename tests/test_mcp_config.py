"""MCP server configuration: file parsing, placeholders, validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.assistant.mcp_config import (
    McpConfigError,
    build_external_servers,
    expand_placeholders,
    load_mcp_file,
    playwright_server,
)
from app.config import Settings


def write(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "mcp_servers.json"
    path.write_text(json.dumps(payload))
    return path


class TestPlaceholders:
    def test_expands_nested_values(self) -> None:
        env = {"TOKEN": "abc"}
        value = {"headers": {"Authorization": "Bearer ${TOKEN}"}, "args": ["${TOKEN}", 3]}
        assert expand_placeholders(value, env) == {
            "headers": {"Authorization": "Bearer abc"},
            "args": ["abc", 3],
        }

    def test_default_and_missing(self) -> None:
        assert expand_placeholders("${NOPE:-fallback}/${ALSO_NOPE}", {}) == "fallback/"


class TestLoadFile:
    def test_missing_file_means_no_servers(self, tmp_path: Path) -> None:
        assert load_mcp_file(tmp_path / "absent.json") == {}

    def test_claude_code_shape(self, tmp_path: Path) -> None:
        path = write(
            tmp_path,
            {
                "mcpServers": {
                    "github": {
                        "type": "http",
                        "url": "https://api.githubcopilot.com/mcp/",
                        "headers": {"Authorization": "Bearer ${GH}"},
                    },
                    "files": {
                        "command": "npx",
                        "args": ["-y", "server-fs", "/tmp"],
                        "env": {"A": 1},
                    },
                }
            },
        )
        servers = load_mcp_file(path, env={"GH": "tok"})
        assert servers["github"] == {
            "type": "http",
            "url": "https://api.githubcopilot.com/mcp/",
            "headers": {"Authorization": "Bearer tok"},
        }
        assert servers["files"] == {
            "type": "stdio",
            "command": "npx",
            "args": ["-y", "server-fs", "/tmp"],
            "env": {"A": "1"},
        }

    def test_flat_shape_is_accepted(self, tmp_path: Path) -> None:
        path = write(tmp_path, {"docs": {"type": "sse", "url": "https://x/sse"}})
        assert load_mcp_file(path)["docs"]["type"] == "sse"

    @pytest.mark.parametrize(
        ("payload", "message"),
        [
            ([], "JSON object"),
            ({"mcpServers": []}, "must be an object"),
            ({"bad name!": {"command": "x"}}, "must match"),
            ({"knowledge": {"command": "x"}}, "reserved"),
            ({"a": "string"}, "must be an object"),
            ({"a": {"type": "stdio"}}, "needs a 'command'"),
            ({"a": {"type": "http"}}, "needs a 'url'"),
            ({"a": {"type": "carrier-pigeon"}}, "unsupported type"),
        ],
    )
    def test_invalid_configs(self, tmp_path: Path, payload: object, message: str) -> None:
        with pytest.raises(McpConfigError, match=message):
            load_mcp_file(write(tmp_path, payload))

    def test_invalid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.json"
        path.write_text("{not json")
        with pytest.raises(McpConfigError, match="not valid JSON"):
            load_mcp_file(path)


class TestBuildServers:
    def test_playwright_command_is_split(self, settings: Settings) -> None:
        server = playwright_server(settings)
        assert server == {
            "type": "stdio",
            "command": "npx",
            "args": ["-y", "@playwright/mcp@latest", "--headless", "--isolated"],
        }

    def test_empty_playwright_command(self, settings: Settings) -> None:
        with pytest.raises(McpConfigError):
            playwright_server(settings.model_copy(update={"agent_playwright_command": " "}))

    def test_merges_playwright_and_file(self, settings: Settings, tmp_path: Path) -> None:
        path = write(tmp_path, {"mcpServers": {"notion": {"type": "http", "url": "https://n"}}})
        servers = build_external_servers(
            settings.model_copy(
                update={"agent_enable_playwright": True, "agent_mcp_config_path": str(path)}
            )
        )
        assert set(servers) == {"playwright", "notion"}

    def test_nothing_enabled(self, settings: Settings) -> None:
        assert build_external_servers(settings) == {}
