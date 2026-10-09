"""GET /jobs/{job_id} -- job state, plus a presigned S3 URL once complete.

boto3-only for the same cold-start reason as api/intake/handler.py.
"""
import json
import os
import uuid

import boto3
from botocore.config import Config

_dynamodb = boto3.resource("dynamodb")
# SigV4 explicitly: unconfigured boto3 presigns with SigV2, which signs
# Content-Type, so a PUT that sends one gets a 403 (and SigV2 is deprecated).
_s3 = boto3.client("s3", config=Config(signature_version="s3v4"))

RESULT_URL_TTL_SEC = 3600
PUBLIC_FIELDS = ("job_id", "status", "source", "match_id", "youtube_url", "error", "progress",
                 "created_at", "updated_at")


def _response(status: int, body: dict) -> dict:
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def handler(event, context):
    job_id = (event.get("pathParameters") or {}).get("job_id", "")
    try:
        uuid.UUID(job_id)
    except ValueError:
        return _response(400, {"error": "job_id must be a UUID"})

    item = _dynamodb.Table(os.environ["DYNAMODB_TABLE_NAME"]).get_item(
        Key={"job_id": job_id},
    ).get("Item")
    if not item:
        return _response(404, {"error": "job not found"})

    body = {k: item[k] for k in PUBLIC_FIELDS if k in item}
    if item.get("video_key"):
        # an uploaded video: the UI plays it back next to the analysis
        body["video_url"] = _s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": os.environ["S3_BUCKET_NAME"], "Key": item["video_key"]},
            ExpiresIn=RESULT_URL_TTL_SEC,
        )
    if item.get("result_key"):
        # The bucket stays private; the client gets a short-lived read link.
        # While processing, the worker rewrites this object after every rally,
        # so the UI can show finished rallies before the whole job is done.
        body["result_url"] = _s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": os.environ["S3_BUCKET_NAME"], "Key": item["result_key"]},
            ExpiresIn=RESULT_URL_TTL_SEC,
        )
    return _response(200, body)
