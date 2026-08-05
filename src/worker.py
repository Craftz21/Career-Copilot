"""
Celery application instance.
Import this module to get the configured Celery app.
All task modules must be listed in `include` so Celery discovers them.
"""

import sys

import structlog
from celery import Celery
from celery.signals import worker_ready

from src.config import get_settings

log = structlog.get_logger(__name__)

settings = get_settings()

celery_app = Celery(
    "career_copilot",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "src.tasks.analyze_resume",
        "src.tasks.analyze_jd",
        "src.tasks.privacy",
    ],
)

celery_app.conf.update(
    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Reliability: only acknowledge a task after it completes successfully.
    # If the worker crashes mid-task, the task is re-queued automatically.
    task_acks_late=True,
    task_reject_on_worker_lost=True,

    # Result TTL: keep task results for 1 hour, then expire
    result_expires=3600,

    # Routing: all tasks go to the default queue for now.
    # Add priority queues here when needed.
    task_default_queue="default",

    # Timezone
    timezone="UTC",
    enable_utc=True,

    # Worker settings
    worker_prefetch_multiplier=1,    # one task at a time per worker slot
    worker_max_tasks_per_child=50,   # restart worker after 50 tasks (prevents memory leaks)
)

# Windows: prefork pool uses spawn (not fork). Spawned child processes start
# with an empty _task_stack, causing fast_trace_task to fail with:
#   ValueError: not enough values to unpack (expected 3, got 0)
# solo pool runs tasks in the main process — no subprocess spawning, no crash.
# On Linux/macOS (Render production) this block is skipped; prefork is used.
if sys.platform == "win32":
    celery_app.conf.worker_pool = "solo"


# ── O8: Pre-warm embedding model on worker startup ────────────────────────────
# Loading sentence-transformers the first time takes 2–10 s on Render's free
# tier. Running it at worker_ready moves that cost to startup (before any
# request), eliminating cold-start latency on the first task.

@worker_ready.connect
def _warm_up_models(sender, **kwargs):
    """Load embedding models into memory immediately after the worker starts."""
    try:
        log.info("worker_model_warmup_start")
        from src.services.skill_extractor import _get_embedding_model as _se_model
        from src.services.role_normalizer import _get_embedding_model as _rn_model
        _se_model()
        _rn_model()
        log.info("worker_model_warmup_done")
    except Exception as exc:
        # Non-fatal: task will load on first invocation instead
        log.warning("worker_model_warmup_failed", error=str(exc))
