"""
Request middleware: structured logging, request ID injection, timing.
"""

import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

log = structlog.get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())[:8]
        t0 = time.monotonic()

        # Inject into request state for downstream access
        request.state.request_id = request_id

        response = await call_next(request)

        duration_ms = int((time.monotonic() - t0) * 1000)

        # Skip logging for static files and health checks
        path = request.url.path
        if not path.startswith("/static") and path != "/health":
            log.info(
                "http_request",
                method=request.method,
                path=path,
                status=response.status_code,
                duration_ms=duration_ms,
                request_id=request_id,
            )

        response.headers["X-Request-ID"] = request_id
        return response
