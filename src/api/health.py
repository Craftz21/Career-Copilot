"""
Health check endpoint for Render uptime monitoring and UptimeRobot keep-warm pings.

GET /health → {"status": "ok"|"degraded", "db": "ok"|"degraded", "redis": "ok"|"degraded"}

Returns 200 only when both database and Redis are reachable.
Returns 503 when either dependency is down so Render marks the instance unhealthy.
"""

import structlog
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from src.config import get_settings
from src.database import get_db

log = structlog.get_logger(__name__)
router = APIRouter(tags=["health"])
settings = get_settings()


@router.get("/health")
async def health_check():
    db_status = "ok"
    redis_status = "ok"

    try:
        with get_db() as db:
            db.execute(text("SELECT 1"))
    except Exception as exc:
        log.warning("health_check_db_fail", error=str(exc))
        db_status = "degraded"

    try:
        import ssl as _ssl
        import redis as _redis
        _url = settings.redis_url.split("?")[0]
        _kwargs = dict(socket_connect_timeout=2, socket_timeout=2)
        if _url.startswith("rediss://"):
            _kwargs["ssl_cert_reqs"] = _ssl.CERT_NONE
        r = _redis.from_url(_url, **_kwargs)
        r.ping()
    except Exception as exc:
        log.warning("health_check_redis_fail", error=str(exc))
        redis_status = "degraded"

    status_code = 200 if db_status == "ok" and redis_status == "ok" else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ok" if status_code == 200 else "degraded",
            "db": db_status,
            "redis": redis_status,
            "version": "2.0.0",
        },
    )
