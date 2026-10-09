"""End-to-end AWS flow against moto (in-memory AWS): intake Lambda ->
DynamoDB + SQS -> worker -> S3 -> status Lambda. No real AWS account or
model calls; analyze_match is stubbed."""
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from shared.models import AnalysisResult, JobResult, RallyReport  # noqa: E402

MATCH_ID = "An_Se_Young_Pornpawee_Chochuwong_TOYOTA_THAILAND_OPEN_2021_QuarterFinals"
URL = "https://www.youtube.com/watch?v=TXT-qlniM90"


def _load_handler(name: str):
    # Both Lambdas are called handler.py, so load each by path under its own name.
    spec = importlib.util.spec_from_file_location(f"{name}_handler", REPO_ROOT / "api" / name / "handler.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def aws(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    with mock_aws():
        boto3.client("s3").create_bucket(Bucket="shuttlecast-test")
        boto3.client("dynamodb").create_table(
            TableName="jobs",
            KeySchema=[{"AttributeName": "job_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "job_id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        queue_url = boto3.client("sqs").create_queue(QueueName="jobs")["QueueUrl"]
        monkeypatch.setenv("S3_BUCKET_NAME", "shuttlecast-test")
        monkeypatch.setenv("DYNAMODB_TABLE_NAME", "jobs")
        monkeypatch.setenv("SQS_QUEUE_URL", queue_url)
        yield
    # handlers bind boto3 clients at import; drop them so the next mock gets fresh ones
    for mod in ("intake_handler", "status_handler"):
        sys.modules.pop(mod, None)


def _post(intake, body) -> dict:
    resp = intake.handler({"body": json.dumps(body)}, None)
    return {**json.loads(resp["body"]), "http": resp["statusCode"]}


def _get(status_mod, job_id) -> dict:
    resp = status_mod.handler({"pathParameters": {"job_id": job_id}}, None)
    return {**json.loads(resp["body"]), "http": resp["statusCode"]}


def _fake_result(job_id: str) -> JobResult:
    analysis = AnalysisResult(
        match_id=MATCH_ID, set_num=1, rally_id=1, depth="deep",
        tactical_pattern="B's cross-court net shot was too high.",
        suggestion_type="shot_selection", suggestion_text="Play a tighter net shot.",
    )
    return JobResult(
        job_id=job_id, match_id=MATCH_ID, youtube_url=URL, fps=30.0,
        rallies=[RallyReport(analysis=analysis, start_time_sec=347.3, rally_winner="A",
                             n_strokes=8, score_a=1, score_b=0)],
    )


# ---- intake -------------------------------------------------------------

def test_intake_creates_job_and_enqueues(aws):
    intake = _load_handler("intake")
    resp = _post(intake, {"match_id": MATCH_ID, "youtube_url": URL})
    assert resp["http"] == 202
    job_id = resp["job_id"]

    item = boto3.resource("dynamodb").Table("jobs").get_item(Key={"job_id": job_id})["Item"]
    assert item["status"] == "queued" and item["match_id"] == MATCH_ID

    from worker.infra import queue
    msg = queue.receive_job(wait_seconds=0)
    assert msg is not None and msg.job_id == job_id


@pytest.mark.parametrize("body,expected", [
    ({"match_id": MATCH_ID, "youtube_url": "https://evil.example.com/x"}, "YouTube"),
    ({"match_id": MATCH_ID, "youtube_url": "http://www.youtube.com/watch?v=x"}, "https"),
    ({"match_id": "", "youtube_url": URL}, "match_id"),
    ({}, "provide match_id"),
    ({"video_key": "../etc/passwd.mp4"}, "uploads/"),
    ([1, 2], "JSON object"),
])
def test_intake_rejects_bad_input(aws, body, expected):
    intake = _load_handler("intake")
    resp = _post(intake, body)
    assert resp["http"] == 400 and expected in resp["error"]


def test_intake_rejects_unknown_match(aws):
    intake = _load_handler("intake")
    intake.KNOWN_MATCHES = {MATCH_ID}
    resp = _post(intake, {"match_id": "Not_A_Real_Match", "youtube_url": URL})
    assert resp["http"] == 400 and "unknown match_id" in resp["error"]


def test_intake_rejects_malformed_json(aws):
    intake = _load_handler("intake")
    resp = intake.handler({"body": "{not json"}, None)
    assert resp["statusCode"] == 400


# ---- status -------------------------------------------------------------

def test_status_404_and_400(aws):
    status_mod = _load_handler("status")
    assert _get(status_mod, "not-a-uuid")["http"] == 400
    assert _get(status_mod, "00000000-0000-0000-0000-000000000000")["http"] == 404


# ---- worker -------------------------------------------------------------

def test_worker_happy_path(aws, monkeypatch, tmp_path):
    from worker import main as worker_main
    from worker.infra import queue, storage

    intake = _load_handler("intake")
    status_mod = _load_handler("status")
    job_id = _post(intake, {"match_id": MATCH_ID, "youtube_url": URL})["job_id"]

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not really a video")
    storage.upload_file(video, storage.video_key(MATCH_ID))

    seen = {}

    real_signature = inspect.signature(worker_main.analyze_match)

    def fake_analyze_match(match_id, video_path, **kwargs):
        # a fake that accepts any kwargs once hid a worker/agent signature
        # mismatch -- bind against the real function so that can't recur
        real_signature.bind(match_id, video_path, **kwargs)
        seen["video_exists"] = Path(video_path).exists()
        kwargs["on_rally_done"](None)  # exercise the visibility heartbeat
        kwargs["on_progress"](_fake_result(kwargs["job_id"]), 4)
        seen["mid_job"] = _get(status_mod, kwargs["job_id"])
        return _fake_result(kwargs["job_id"])

    monkeypatch.setattr(worker_main, "analyze_match", fake_analyze_match)

    worker_main.handle(queue.receive_job(wait_seconds=0))

    assert seen["video_exists"], "worker should have pulled the clip from S3"
    mid = seen["mid_job"]  # finished rallies are readable before the job completes
    assert mid["status"] == "processing" and mid["progress"] == "Analyzed 1 of 4 rallies"
    assert mid["result_url"].startswith("https://")
    resp = _get(status_mod, job_id)
    assert resp["http"] == 200 and resp["status"] == "complete"
    assert resp["result_url"].startswith("https://")
    saved = JobResult(**storage.get_json(storage.result_key(job_id)))
    assert saved.rallies[0].analysis.suggestion_type == "shot_selection"
    assert queue.receive_job(wait_seconds=0) is None, "message should be deleted after success"


def test_worker_retries_then_fails(aws, monkeypatch, tmp_path):
    from worker import main as worker_main
    from worker.infra import queue, state, storage

    intake = _load_handler("intake")
    job_id = _post(intake, {"match_id": MATCH_ID, "youtube_url": URL})["job_id"]
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    storage.upload_file(video, storage.video_key(MATCH_ID))

    def boom(*args, **kwargs):
        raise RuntimeError("groq exploded")

    monkeypatch.setattr(worker_main, "analyze_match", boom)

    msg = queue.receive_job(wait_seconds=0)
    worker_main.handle(msg)  # attempt 1: back to queued, message kept
    job = state.get_job(job_id)
    assert job.status == "queued" and "attempt 1 failed" in job.error

    msg.receive_count = worker_main.MAX_ATTEMPTS  # simulate the final redelivery
    worker_main.handle(msg)
    job = state.get_job(job_id)
    assert job.status == "failed" and "groq exploded" in job.error


def test_worker_ignores_duplicate_delivery(aws, monkeypatch):
    from worker import main as worker_main
    from worker.infra import queue, state

    intake = _load_handler("intake")
    job_id = _post(intake, {"match_id": MATCH_ID, "youtube_url": URL})["job_id"]
    state.update_status(job_id, "complete", result_key="results/x.json")

    monkeypatch.setattr(worker_main, "analyze_match", lambda *a, **k: pytest.fail("must not reprocess"))
    worker_main.handle(queue.receive_job(wait_seconds=0))
    assert queue.receive_job(wait_seconds=0) is None


def test_update_status_refuses_phantom_job(aws):
    from botocore.exceptions import ClientError
    from worker.infra import state

    with pytest.raises(ClientError):
        state.update_status("00000000-0000-0000-0000-000000000000", "processing")


# ---- any-video jobs (YouTube link or upload) ------------------------------

def test_youtube_only_creates_video_job(aws):
    intake = _load_handler("intake")
    resp = _post(intake, {"youtube_url": URL})
    assert resp["http"] == 202
    item = boto3.resource("dynamodb").Table("jobs").get_item(Key={"job_id": resp["job_id"]})["Item"]
    assert item["source"] == "video" and "match_id" not in item


def test_upload_flow_presign_then_submit(aws):
    intake = _load_handler("intake")
    status_mod = _load_handler("status")
    up = intake.handler({"routeKey": "POST /uploads", "body": json.dumps({"filename": "rally.mp4"})}, None)
    body = json.loads(up["body"])
    assert up["statusCode"] == 200 and body["video_key"].startswith("uploads/")
    assert "X-Amz-Signature" in body["upload_url"] or "Signature" in body["upload_url"]

    # submitting before the file exists is rejected
    early = _post(intake, {"video_key": body["video_key"]})
    assert early["http"] == 400 and "no uploaded video" in early["error"]

    boto3.client("s3").put_object(Bucket="shuttlecast-test", Key=body["video_key"], Body=b"fake")
    job = _post(intake, {"video_key": body["video_key"]})
    assert job["http"] == 202
    st = _get(status_mod, job["job_id"])
    assert st["source"] == "video" and st["video_url"].startswith("https://")


def test_upload_rejects_non_video(aws):
    intake = _load_handler("intake")
    up = intake.handler({"routeKey": "POST /uploads", "body": json.dumps({"filename": "notes.exe"})}, None)
    assert up["statusCode"] == 400


def test_worker_video_job_without_rallies_fails_once(aws, monkeypatch):
    """No rallies in the footage is permanent: fail now, don't retry 3 times."""
    from worker import main as worker_main
    from worker.infra import queue, state

    intake = _load_handler("intake")
    s3 = boto3.client("s3")
    up = json.loads(intake.handler({"routeKey": "POST /uploads", "body": json.dumps({"filename": "x.mp4"})}, None)["body"])
    s3.put_object(Bucket="shuttlecast-test", Key=up["video_key"], Body=b"fake")
    job_id = _post(intake, {"video_key": up["video_key"]})["job_id"]

    class NoRallies:
        def run(self, *a, **k):
            return [], 30.0

    monkeypatch.setattr(worker_main, "recognizer", lambda: NoRallies())
    msg = queue.receive_job(wait_seconds=0)
    assert msg.receive_count == 1
    worker_main.handle(msg)
    job = state.get_job(job_id)
    assert job.status == "failed" and "No rallies found" in job.error
    assert queue.receive_job(wait_seconds=0) is None, "permanent failure must not be retried"
