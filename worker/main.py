"""Fargate worker entrypoint: SQS -> analyze match -> S3 result + DynamoDB state.

Run: python -m worker.main
"""
import os
import signal
import tempfile
from pathlib import Path

from shared.models import Job, JobResult
from worker.agent.react_loop import analyze_match, analyze_rallies
from worker.infra import queue, state, storage
from worker.observability.logger import bind, clear, log_event, timed
from worker.pipeline.ingest import download_video

# Kept below the queue's DLQ maxReceiveCount (5, infra/template.yaml) so the
# worker marks a job failed itself before SQS quietly moves it to the DLQ.
MAX_ATTEMPTS = int(os.environ.get("MAX_ATTEMPTS", "3"))
VISIBILITY_HEARTBEAT_SEC = 900
RETRY_DELAY_SEC = 60
MAX_RALLIES = int(os.environ["MAX_RALLIES"]) if os.environ.get("MAX_RALLIES") else None
MAX_VIDEO_MINUTES = float(os.environ.get("MAX_VIDEO_MINUTES", "10"))


class PermanentJobError(Exception):
    """Retrying can't help (no rallies in the video, YouTube refused the
    download) -- fail the job now with a message the user can act on."""


def _explain_rally_failures(result: JobResult) -> str:
    first = result.errors[0].error
    if "tokens per day" in first:
        # Groq's free tier: 200K tokens/day per model, on a rolling 24h window.
        # An SQS retry minutes later would hit the same wall.
        return ("The AI model's free daily token quota (Groq) is used up, so no rally "
                "could be analyzed. Try again later.")
    return f"All {len(result.errors)} rallies failed to analyze. First error: {first[:300]}"


_recognizer = None


def recognizer():
    """Load the stroke models once per worker, not per job (and only when a
    video job actually arrives)."""
    global _recognizer
    if _recognizer is None:
        from vision.infer import StrokeRecognizer

        _recognizer = StrokeRecognizer()
    return _recognizer

_shutdown = False


def _on_sigterm(*_) -> None:
    # ECS sends SIGTERM on deploys/scale-in. Finish the current job, take no new one.
    global _shutdown
    _shutdown = True
    log_event("sigterm_received")


def fetch_video(job: Job, work_dir: Path) -> Path:
    """S3 first, YouTube as fallback. YouTube blocks datacenter IPs far more
    aggressively than residential ones, so in practice
    clips are uploaded to videos/{match_id}.mp4 ahead of time."""
    dest = work_dir / f"{job.match_id}.mp4"
    key = storage.video_key(job.match_id)
    if storage.exists(key):
        return storage.download_file(key, dest)
    download_video(job.youtube_url, dest)
    storage.upload_file(dest, key)  # cache it for the next job on this match
    return dest


def analyze_video_job(job: Job, work_dir: Path, heartbeat, on_progress=None):
    """Footage nobody annotated: recover strokes with the trained vision
    models, then run the same agent on each detected rally."""
    dest = work_dir / "video.mp4"
    with timed("video_fetch"):
        if job.video_key:
            storage.download_file(job.video_key, dest)
        else:
            try:
                download_video(job.youtube_url, dest)
            except Exception as exc:
                # YouTube blocks datacenter IPs far more than home connections
                raise PermanentJobError(
                    "YouTube refused the download from the cloud worker. Upload the video file instead."
                ) from exc
    state.update_status(job.job_id, "processing", progress="Finding rallies in the video")
    with timed("stroke_detection"):
        detected, fps = recognizer().run(dest, match_id=f"video-{job.job_id[:8]}", max_minutes=MAX_VIDEO_MINUTES)
    log_event("strokes_detected", rallies=len(detected), strokes=sum(len(d.rally.strokes) for d in detected))
    if not detected:
        raise PermanentJobError(
            "No rallies found. This works on broadcast-style singles footage: a fixed camera "
            "behind the court and a green court mat."
        )
    rallies = [d.rally for d in detected][:MAX_RALLIES] if MAX_RALLIES else [d.rally for d in detected]
    with timed("match_analysis"):
        return analyze_rallies(rallies[0].match_id, rallies, dest, fps, job_id=job.job_id,
                               youtube_url=job.youtube_url, on_rally_done=heartbeat,
                               on_progress=on_progress)


def process(msg: queue.QueueMessage) -> None:
    job = state.get_job(msg.job_id)
    if job is None:
        log_event("job_missing", job_id=msg.job_id)
        queue.delete(msg)
        return
    if job.status == "complete":
        # SQS is at-least-once; a redelivered message for a finished job is a no-op.
        log_event("duplicate_delivery", job_id=job.job_id)
        queue.delete(msg)
        return

    bind(job_id=job.job_id, match_id=job.match_id, source=job.source)
    log_event("job_started", attempt=msg.receive_count)
    state.update_status(job.job_id, "processing", progress="Starting")
    heartbeat = lambda _rally: queue.extend_visibility(msg, VISIBILITY_HEARTBEAT_SEC)  # noqa: E731
    key = storage.result_key(job.job_id)

    def on_progress(partial: JobResult, total: int) -> None:
        # Rewrite the result object after every rally so the UI can show
        # finished rallies while the rest are still running. Status stays
        # 'processing'; only the final write flips it to 'complete'.
        storage.put_json(key, partial.model_dump_json())
        done = len(partial.rallies) + len(partial.errors)
        state.update_status(job.job_id, "processing", result_key=key,
                            progress=f"Analyzed {done} of {total} rallies")

    with tempfile.TemporaryDirectory() as tmp:
        if job.source == "video":
            result = analyze_video_job(job, Path(tmp), heartbeat, on_progress)
        else:
            with timed("video_fetch"):
                video_path = fetch_video(job, Path(tmp))
            with timed("match_analysis"):
                result = analyze_match(
                    job.match_id, video_path,
                    max_rallies=MAX_RALLIES, job_id=job.job_id, youtube_url=job.youtube_url,
                    on_rally_done=heartbeat, on_progress=on_progress,
                )

    if not result.rallies and result.errors:
        # Every rally failed: that's a failed job, not a 'complete' one with
        # nothing in it (which the UI can't render).
        raise PermanentJobError(_explain_rally_failures(result))
    with timed("s3_write"):
        storage.put_json(key, result.model_dump_json())
    state.update_status(job.job_id, "complete", result_key=key)
    queue.delete(msg)
    log_event("job_complete", n_rallies=len(result.rallies), n_rally_errors=len(result.errors))


def handle(msg: queue.QueueMessage) -> None:
    try:
        process(msg)
    except PermanentJobError as exc:
        log_event("job_error", job_id=msg.job_id, permanent=True, error=str(exc))
        state.update_status(msg.job_id, "failed", error=str(exc))
        queue.delete(msg)
    except Exception as exc:
        log_event("job_error", job_id=msg.job_id, attempt=msg.receive_count, error=repr(exc)[:500])
        if msg.receive_count >= MAX_ATTEMPTS:
            state.update_status(msg.job_id, "failed", error=repr(exc))
            queue.delete(msg)
        else:
            state.update_status(
                msg.job_id, "queued", error=f"attempt {msg.receive_count} failed, retrying: {exc!r}",
            )
            # back off before SQS redelivers it
            queue.extend_visibility(msg, RETRY_DELAY_SEC)
    finally:
        clear()


def main() -> None:
    signal.signal(signal.SIGTERM, _on_sigterm)
    log_event("worker_started", max_attempts=MAX_ATTEMPTS, max_rallies=MAX_RALLIES)
    while not _shutdown:
        msg = queue.receive_job()
        if msg is not None:
            handle(msg)
    log_event("worker_stopped")


if __name__ == "__main__":
    main()
