"""S3 helpers. Layout in the bucket:

    videos/{match_id}.mp4     input clips (pre-uploaded; see worker/main.py)
    results/{job_id}.json     JobResult output
"""
import json
import os
from pathlib import Path

import boto3
from botocore.exceptions import ClientError


def _bucket() -> str:
    return os.environ["S3_BUCKET_NAME"]


def _client():
    return boto3.client("s3")


def video_key(match_id: str) -> str:
    return f"videos/{match_id}.mp4"


def result_key(job_id: str) -> str:
    return f"results/{job_id}.json"


def exists(key: str) -> bool:
    try:
        _client().head_object(Bucket=_bucket(), Key=key)
        return True
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def download_file(key: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    _client().download_file(_bucket(), key, str(dest))
    return dest


def upload_file(src: Path, key: str) -> None:
    _client().upload_file(str(src), _bucket(), key)


def put_json(key: str, body: str | dict) -> None:
    data = body if isinstance(body, str) else json.dumps(body)
    _client().put_object(
        Bucket=_bucket(), Key=key, Body=data.encode("utf-8"),
        ContentType="application/json",
    )


def get_json(key: str) -> dict:
    obj = _client().get_object(Bucket=_bucket(), Key=key)
    return json.loads(obj["Body"].read())
