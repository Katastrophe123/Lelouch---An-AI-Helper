"""Assemble the MCP servers the agent can use.

Three sources, merged in this order (later names win):

1. **Playwright** (``AGENT_ENABLE_PLAYWRIGHT``): a real browser the agent can
   drive - navigate, click, type, read pages, take screenshots.
2. **Your MCP servers** from ``AGENT_MCP_CONFIG_PATH``, in the same JSON shape
   as Claude Code's ``.mcp.json``::

       {"mcpServers": {"github": {"type": "http", "url": "...",
                                  "headers": {"Authorization": "Bearer ${GITHUB_TOKEN}"}}}}

   ``${VAR}`` placeholders are filled from the environment so secrets stay in
   ``.env``, not in the file.
3. **knowledge** (in-process): ``search_my_documents`` over your uploads. It is
   added by the runner, not here, because it is bound to one user per turn.
"""

from __future__ import annotations

import json
import os
import re
import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from app.config import Settings
from app.logging_config import get_logger

logger = get_logger(__name__)

_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
_VALID_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
RESERVED_NAMES = frozenset({"knowledge"})


class McpConfigError(ValueError):
    """The MCP config file is malformed."""


def expand_placeholders(value: Any, env: Mapping[str, str] | None = None) -> Any:
    """Recursively replace ``${VAR}`` / ``${VAR:-default}`` in strings."""
    source: Mapping[str, str] = os.environ if env is None else env
    if isinstance(value, str):
        return _PLACEHOLDER.sub(lambda m: source.get(m.group(1), m.group(2) or ""), value)
    if isinstance(value, list):
        return [expand_placeholders(item, source) for item in value]
    if isinstance(value, dict):
        return {key: expand_placeholders(item, source) for key, item in value.items()}
    return value


def _validate_server(name: str, server: Any) -> dict[str, Any]:
    if not _VALID_NAME.match(name):
        raise McpConfigError(f"MCP server name {name!r} must match {_VALID_NAME.pattern}")
    if name in RESERVED_NAMES:
        raise McpConfigError(f"MCP server name {name!r} is reserved")
    if not isinstance(server, dict):
        raise McpConfigError(f"MCP server {name!r} must be an object")
    kind = server.get("type", "stdio")
    if kind == "stdio":
        if not isinstance(server.get("command"), str) or not server["command"]:
            raise McpConfigError(f"stdio MCP server {name!r} needs a 'command'")
        config: dict[str, Any] = {"type": "stdio", "command": server["command"]}
        if server.get("args") is not None:
            config["args"] = [str(arg) for arg in server["args"]]
        if server.get("env"):
            config["env"] = {str(k): str(v) for k, v in dict(server["env"]).items()}
        return config
    if kind in ("http", "sse"):
        if not isinstance(server.get("url"), str) or not server["url"]:
            raise McpConfigError(f"{kind} MCP server {name!r} needs a 'url'")
        config = {"type": kind, "url": server["url"]}
        if server.get("headers"):
            config["headers"] = {str(k): str(v) for k, v in dict(server["headers"]).items()}
        return config
    raise McpConfigError(f"MCP server {name!r} has unsupported type {kind!r}")


def load_mcp_file(path: str | Path, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Parse an ``.mcp.json``-style file. A missing file means no servers."""
    file = Path(path)
    if not file.exists():
        return {}
    try:
        raw = json.loads(file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise McpConfigError(f"{file} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise McpConfigError(f"{file} must contain a JSON object")
    servers = raw.get("mcpServers", raw)
    if not isinstance(servers, dict):
        raise McpConfigError(f"'mcpServers' in {file} must be an object")
    expanded = expand_placeholders(servers, env)
    return {name: _validate_server(name, server) for name, server in expanded.items()}


def playwright_server(settings: Settings) -> dict[str, Any]:
    parts = shlex.split(settings.agent_playwright_command)
    if not parts:
        raise McpConfigError("AGENT_PLAYWRIGHT_COMMAND is empty")
    return {"type": "stdio", "command": parts[0], "args": parts[1:]}


def build_external_servers(settings: Settings) -> dict[str, Any]:
    """Every MCP server except the per-user ``knowledge`` one."""
    servers: dict[str, Any] = {}
    if settings.agent_enable_playwright:
        servers["playwright"] = playwright_server(settings)
    if settings.agent_mcp_config_path:
        servers.update(load_mcp_file(settings.agent_mcp_config_path))
    return servers
