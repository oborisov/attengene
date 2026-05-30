"""
API key authentication for AttenGene.

When ATTENGENE_API_KEY is set, requires Bearer token in Authorization header.
When not set (development), auth is disabled.
"""

import os

from fastapi import HTTPException, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.audit import log_auth_event

_API_KEY = os.getenv("ATTENGENE_API_KEY", "")
_security = HTTPBearer(auto_error=False)


async def verify_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Security(_security),
) -> None:
    """
    Verify API key from Authorization: Bearer header.

    Raises HTTPException 401 if key is required but missing/wrong.
    No-op when ATTENGENE_API_KEY is not set (dev mode).
    """
    if not _API_KEY:
        return

    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")
    endpoint = request.url.path

    if credentials is None:
        log_auth_event("api_key_missing", client_ip, user_agent, endpoint)
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    if credentials.credentials != _API_KEY:
        log_auth_event("api_key_failure", client_ip, user_agent, endpoint)
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    log_auth_event("api_key_success", client_ip, user_agent, endpoint)
