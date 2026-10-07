"""Structured JSON logs -> stdout -> CloudWatch.

On Fargate the awslogs log driver ships stdout to CloudWatch, so one JSON
object per line is all it takes for Logs Insights to query fields
directly. p95 latency per pipeline stage, for example:

    fields stage, duration_ms
    | filter event = "stage_end" and status = "ok"
    | stats pct(duration_ms, 95) as p95_ms, count(*) as n by stage
"""
import json
import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone

_context: ContextVar[dict] = ContextVar("log_context", default={})


def bind(**fields) -> None:
    """Attach fields (job_id, match_id, ...) to every later log line."""
    _context.set({**_context.get(), **fields})


def clear() -> None:
    _context.set({})


def log_event(event: str, **fields) -> None:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **_context.get(),
        **fields,
    }
    print(json.dumps(record, default=str), file=sys.stdout, flush=True)


@contextmanager
def timed(stage: str, **fields):
    """Log a stage_end line with duration_ms and ok/error status."""
    start = time.perf_counter()
    try:
        yield
    except Exception as exc:
        log_event(
            "stage_end", stage=stage, status="error",
            duration_ms=round((time.perf_counter() - start) * 1000, 1),
            error=repr(exc)[:500], **fields,
        )
        raise
    log_event(
        "stage_end", stage=stage, status="ok",
        duration_ms=round((time.perf_counter() - start) * 1000, 1), **fields,
    )
