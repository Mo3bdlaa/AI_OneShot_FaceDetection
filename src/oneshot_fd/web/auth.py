"""Access control for the web interface.

The UI can enrol people, delete reference photos and switch on a camera. On
``127.0.0.1`` that is nobody's business but the user's, and a password would
only be in the way. The moment it is bound to ``0.0.0.0`` it is a different
proposition: everyone on the network can do all of that.

So: a token is required whenever the server is reachable from outside this
machine, and one is generated if the user did not supply one. Nothing here
protects the *stream* from someone who already has the token - it is a shared
secret for a tool on your own network, not an authentication system.
"""

from __future__ import annotations

import hmac
import os
import secrets
from typing import Optional

from ..utils import LOGGER

#: Read when no token is passed on the command line. Containers set this so
#: `docker compose up` is predictable instead of printing a fresh secret.
TOKEN_ENV = "ONESHOT_TOKEN"

#: Paths that stay open, so a container health check needs no credentials.
PUBLIC_PATHS = frozenset({"/api/health", "/favicon.ico"})

LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def resolve_token(token: Optional[str], host: str) -> Optional[str]:
    """Decide what token (if any) this server should require.

    An explicit token is always honoured. Otherwise a token is generated when
    the server is exposed beyond this machine, and skipped when it is not.
    """
    if token:
        return token
    from_env = os.environ.get(TOKEN_ENV, "").strip()
    if from_env:
        LOGGER.info("Using the token from $%s.", TOKEN_ENV)
        return from_env
    if host in LOCAL_HOSTS:
        return None
    generated = secrets.token_urlsafe(12)
    LOGGER.warning(
        "Serving on %s, which other machines can reach, so access needs a token.\n"
        "        Open:  http://<this-machine>:<port>/?token=%s\n"
        "        Pass --token or set $%s to choose your own, "
        "or --no-token to turn this off.",
        host, generated, TOKEN_ENV,
    )
    return generated


def token_matches(expected: Optional[str], supplied: Optional[str]) -> bool:
    """Constant-time comparison, so the check cannot be timed character by character."""
    if not expected:
        return True
    if not supplied:
        return False
    return hmac.compare_digest(str(expected), str(supplied))


def install(app, token: Optional[str]) -> None:
    """Add the token check to ``app``. A ``None`` token installs nothing."""
    if not token:
        return

    from fastapi import Request
    from fastapi.responses import JSONResponse

    @app.middleware("http")
    async def require_token(request: Request, call_next):
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        supplied = (
            request.headers.get("x-token")
            or request.query_params.get("token")
            or request.cookies.get("oneshot_token")
        )
        if not token_matches(token, supplied):
            return JSONResponse(
                status_code=401,
                content={"detail": "This server needs a token. Open it with ?token=…"},
            )

        response = await call_next(request)
        # Remember it, so the <img> stream and later calls do not each need it
        # spelled out in the URL.
        if request.query_params.get("token"):
            response.set_cookie("oneshot_token", token, httponly=True, samesite="strict")
        return response
