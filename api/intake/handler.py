"""POST /jobs -- validate, create job, enqueue, return job_id immediately.

Deliberately boto3-only (no shared/ or pydantic): the Lambda zip stays tiny
and cold starts stay fast. The DynamoDB item mirrors shared.models.Job.
"""
import base64
import csv
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import boto3

# Module scope so warm invocations reuse the clients instead of rebuilding them.
_dynamodb = boto3.resource("dynamodb")
_sqs = boto3.client("sqs")

YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}

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


def validate(payload) -> tuple[dict | None, str | None]:
    if not isinstance(payload, dict):
        return None, "body must be a JSON object"
    match_id = payload.get("match_id")
    url = payload.get("youtube_url")
    if not isinstance(match_id, str) or not match_id.strip() or len(match_id) > 200:
        return None, "match_id must be a non-empty string"
    if KNOWN_MATCHES is not None and match_id not in KNOWN_MATCHES:
        return None, f"unknown match_id: {match_id}"
    if not isinstance(url, str) or len(url) > 500:
        return None, "youtube_url must be a string"
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in YOUTUBE_HOSTS:
        return None, "youtube_url must be an https YouTube link"
    return {"match_id": match_id, "youtube_url": url}, None


def handler(event, context):
    raw = event.get("body") or ""
    if event.get("isBase64Encoded"):
        raw = base64.b64decode(raw).decode("utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return _response(400, {"error": "body must be valid JSON"})

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
