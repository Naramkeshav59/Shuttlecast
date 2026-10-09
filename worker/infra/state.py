"""DynamoDB job state: job_id -> Job. Partition key `job_id` (string)."""
import os
from datetime import datetime, timezone

import boto3

from shared.models import Job


def _table():
    return boto3.resource("dynamodb").Table(os.environ["DYNAMODB_TABLE_NAME"])


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_job(job: Job) -> Job:
    now = _now()
    job = job.model_copy(update={"created_at": job.created_at or now, "updated_at": now})
    _table().put_item(Item=job.model_dump(exclude_none=True))
    return job


def get_job(job_id: str) -> Job | None:
    item = _table().get_item(Key={"job_id": job_id}).get("Item")
    return Job(**item) if item else None


def update_status(
    job_id: str, status: str, *, result_key: str | None = None, error: str | None = None,
    progress: str | None = None,
) -> None:
    names = {"#s": "status"}
    values = {":s": status, ":u": _now()}
    sets = ["#s = :s", "updated_at = :u"]
    if progress is not None:
        sets.append("progress = :p")
        values[":p"] = progress
    if result_key is not None:
        sets.append("result_key = :r")
        values[":r"] = result_key
    if error is not None:
        sets.append("#e = :e")
        names["#e"] = "error"
        values[":e"] = error[:1000]
    _table().update_item(
        Key={"job_id": job_id},
        UpdateExpression="SET " + ", ".join(sets),
        ExpressionAttributeNames=names,
        ExpressionAttributeValues=values,
        # never create a phantom item from a stale/forged job_id
        ConditionExpression="attribute_exists(job_id)",
    )
