"""Token issuing endpoint.

Identity model: ``username`` names the learner and is created on first
sign-in; ``password`` is a single shared application password from the
environment. Deliberately minimal - see README "Limitations".
"""

from __future__ import annotations

import hmac

from fastapi import APIRouter, status

from app.api.deps import SettingsDep, UserRepositoryDep
from app.api.errors import UnauthorizedError
from app.core.security import create_access_token
from app.logging_config import get_logger
from app.models.schemas import TokenRequest, TokenResponse

logger = get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/token",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="Sign in (creates the learner on first use)",
    responses={401: {"description": "Invalid password"}},
)
async def issue_token(
    payload: TokenRequest,
    settings: SettingsDep,
    users: UserRepositoryDep,
) -> TokenResponse:
    if not hmac.compare_digest(payload.password, settings.auth_password):
        logger.warning("auth_failed", username=payload.username)
        raise UnauthorizedError("Incorrect password")

    user = await users.get_or_create(payload.username)
    token, expires_in = create_access_token(
        user.username, settings, extra_claims={"uid": str(user.id)}
    )
    logger.info("auth_token_issued", username=user.username, expires_in=expires_in)
    return TokenResponse(
        access_token=token,
        expires_in=expires_in,
        user_id=user.id,
        username=user.username,
    )
