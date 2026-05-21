"""Session helpers — both apps hold a JWT in a session cookie."""

from fastapi import Cookie, HTTPException


async def get_current_user(session_token: str | None = Cookie(default=None)):
    if not session_token:
        raise HTTPException(401, "not_authenticated")
    # TODO: validate token and return a User via shared.api_client.
    raise NotImplementedError
