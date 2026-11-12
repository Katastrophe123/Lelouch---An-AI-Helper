"""Agent capabilities, for clients that render a model picker."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import AgentServiceDep, CurrentUserDep, SettingsDep
from app.assistant.runner import auth_mode
from app.models.schemas import AgentConfigResponse

router = APIRouter(prefix="/agent", tags=["agent"])


@router.get("/config", response_model=AgentConfigResponse, summary="What the agent can use")
async def agent_config(
    _user: CurrentUserDep, settings: SettingsDep, agent: AgentServiceDep
) -> AgentConfigResponse:
    return AgentConfigResponse(
        name=settings.agent_name,
        backend=agent.runner.name,
        auth=auth_mode() if agent.runner.name == "claude_sdk" else "none",
        default_model=settings.agent_model,
        models=list(settings.agent_models),
        effort=settings.agent_effort,
        thinking=settings.agent_thinking,
        tools=agent.runner.tool_names(),
        mcp_servers=agent.runner.mcp_server_names(),
        max_budget_usd=settings.agent_max_budget_usd,
    )
