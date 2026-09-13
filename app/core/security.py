from typing import Optional
from fastapi import Header, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from app.core.config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_current_user_id(
    x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Security(api_key_header),
) -> str:
    """
    Authenticates requests if API_KEY is configured, and extracts or defaults the user/tenant identifier.
    """
    if settings.API_KEY:
        token = None
        if x_api_key:
            token = x_api_key
        elif authorization and authorization.lower().startswith("bearer "):
            token = authorization[7:].strip()

        if token != settings.API_KEY:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    return x_user_id or "default_user"
