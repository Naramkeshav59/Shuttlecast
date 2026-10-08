"""Intake Lambda.

POST /jobs     -- validate, create the job, enqueue, return job_id immediately.
                  Two kinds of job:
                    * ShuttleSet match: {"match_id", "youtube_url"}
                    * any video:        {"youtube_url"}  or  {"video_key"} (an upload)
POST /uploads  -- {"filename"} -> a short-lived presigned S3 PUT URL plus the
                  video_key to submit afterwards. The client uploads straight
                  to S3, so large videos never pass through Lambda (6 MB limit)
                  or API Gateway (10 MB).

Deliberately boto3-only (no shared/ or pydantic): the Lambda zip stays tiny
and cold starts stay fast. The DynamoDB item mirrors shared.models.Job.
"""
import base64
import csv
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError

# Module scope so warm invocations reuse the clients instead of rebuilding them.
_dynamodb = boto3.resource("dynamodb")
_sqs = boto3.client("sqs")
_s3 = boto3.client("s3")

YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm"}
UPLOAD_URL_TTL_SEC = 900

# scripts/deploy_api.sh copies ShuttleSet's match.csv into the zip so bad
# match_ids are rejected at the edge instead of failing minutes later in a worker.
_MATCH_CSV = Path(__file__).with_name("match.csv")


def _load_known_matches() -> set[str] | None:
    if not _MATCH_CSV.exists():
        return None
    with _MATCH_CSV.open(encoding="utf-8") as f:
        return {row["video"] for row in csv.DictReader(f)}


KNOWN_MATCHES = _load_known_matches()


def _response(status: int, body: dict) -> dict:
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def _youtube_ok(url) -> bool:
    if not isinstance(url, str) or len(url) > 500:
        return False
    parsed = urlparse(url)
    return parsed.scheme == "https" and parsed.hostname in YOUTUBE_HOSTS


def _uploaded(key: str) -> bool:
    try:
        _s3.head_object(Bucket=os.environ["S3_BUCKET_NAME"], Key=key)
        return True
    except ClientError:
        return False


def validate(payload) -> tuple[dict | None, str | None]:
    if not isinstance(payload, dict):
        return None, "body must be a JSON object"
    match_id = payload.get("match_id")
    url = payload.get("youtube_url")
    video_key = payload.get("video_key")

    if match_id is not None:  # annotated ShuttleSet match
        if not isinstance(match_id, str) or not match_id.strip() or len(match_id) > 200:
            return None, "match_id must be a non-empty string"
        if KNOWN_MATCHES is not None and match_id not in KNOWN_MATCHES:
            return None, f"unknown match_id: {match_id}"
        if not _youtube_ok(url):
            return None, "youtube_url must be an https YouTube link"
        return {"source": "shuttleset", "match_id": match_id, "youtube_url": url}, None

    if video_key is not None:  # a video the client already uploaded
        if (not isinstance(video_key, str) or not video_key.startswith("uploads/")
                or PurePosixPath(video_key).suffix.lower() not in VIDEO_EXTS or ".." in video_key):
            return None, "video_key must be an uploads/ key returned by POST /uploads"
        if not _uploaded(video_key):
            return None, "no uploaded video at that video_key yet"
        return {"source": "video", "video_key": video_key}, None

    if url is not None:  # any YouTube video
        if not _youtube_ok(url):
            return None, "youtube_url must be an https YouTube link"
        return {"source": "video", "youtube_url": url}, None

    return None, "provide match_id + youtube_url, a youtube_url, or a video_key"


def _body(event):
    raw = event.get("body") or ""
    if event.get("isBase64Encoded"):
        raw = base64.b64decode(raw).decode("utf-8")
    return json.loads(raw)


def create_upload(payload) -> dict:
    name = payload.get("filename") if isinstance(payload, dict) else None
    ext = PurePosixPath(name).suffix.lower() if isinstance(name, str) else ""
    if ext not in VIDEO_EXTS:
        return _response(400, {"error": f"filename must end in one of {sorted(VIDEO_EXTS)}"})
    key = f"uploads/{uuid.uuid4()}{ext}"
    url = _s3.generate_presigned_url(
        "put_object", Params={"Bucket": os.environ["S3_BUCKET_NAME"], "Key": key},
        ExpiresIn=UPLOAD_URL_TTL_SEC,
    )
    return _response(200, {"upload_url": url, "video_key": key, "expires_in": UPLOAD_URL_TTL_SEC})


def create_job(payload) -> dict:
    fields, err = validate(payload)
    if err:
        return _response(400, {"error": err})

    job_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    table = _dynamodb.Table(os.environ["DYNAMODB_TABLE_NAME"])
    # DynamoDB first, SQS second: the worker must never receive a job_id
    # that doesn't exist yet.
    table.put_item(Item={
        "job_id": job_id, "status": "queued", **fields,
        "created_at": now, "updated_at": now,
    })
    try:
        _sqs.send_message(
            QueueUrl=os.environ["SQS_QUEUE_URL"],
            MessageBody=json.dumps({"job_id": job_id}),
        )
    except Exception as exc:
        # otherwise the job sits "queued" forever with nothing to process it
        table.update_item(
            Key={"job_id": job_id},
            UpdateExpression="SET #s = :s, #e = :e",
            ExpressionAttributeNames={"#s": "status", "#e": "error"},
            ExpressionAttributeValues={":s": "failed", ":e": f"enqueue failed: {exc!r}"[:1000]},
        )
        return _response(500, {"error": "could not enqueue job", "job_id": job_id})

    return _response(202, {"job_id": job_id, "status": "queued"})


def handler(event, context):
    try:
        payload = _body(event)
    except json.JSONDecodeError:
        return _response(400, {"error": "body must be valid JSON"})
    if event.get("routeKey") == "POST /uploads":
        return create_upload(payload)
    return create_job(payload)
