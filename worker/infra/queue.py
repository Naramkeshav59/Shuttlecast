"""SQS consumer helpers. Message body is just {"job_id": ...}; DynamoDB is
the source of truth for the job itself."""
import json
import os
from dataclasses import dataclass

import boto3


@dataclass
class QueueMessage:
    job_id: str
    receipt_handle: str
    receive_count: int


def _url() -> str:
    return os.environ["SQS_QUEUE_URL"]


def _client():
    return boto3.client("sqs")


def send_job(job_id: str) -> None:
    _client().send_message(QueueUrl=_url(), MessageBody=json.dumps({"job_id": job_id}))


def receive_job(wait_seconds: int = 20) -> QueueMessage | None:
    """Long-poll for one job. wait_seconds=20 (the SQS max) means an idle
    worker makes ~3 receive calls/min instead of spinning on empty ones."""
    resp = _client().receive_message(
        QueueUrl=_url(),
        MaxNumberOfMessages=1,
        WaitTimeSeconds=wait_seconds,
        MessageSystemAttributeNames=["ApproximateReceiveCount"],
    )
    messages = resp.get("Messages", [])
    if not messages:
        return None
    msg = messages[0]
    return QueueMessage(
        job_id=json.loads(msg["Body"])["job_id"],
        receipt_handle=msg["ReceiptHandle"],
        receive_count=int(msg.get("Attributes", {}).get("ApproximateReceiveCount", 1)),
    )


def extend_visibility(msg: QueueMessage, seconds: int) -> None:
    """Heartbeat: a full match takes far longer than any sane visibility
    timeout, and if it lapses SQS redelivers the job to another worker
    mid-run. Called after every rally."""
    _client().change_message_visibility(
        QueueUrl=_url(), ReceiptHandle=msg.receipt_handle, VisibilityTimeout=seconds,
    )


def delete(msg: QueueMessage) -> None:
    _client().delete_message(QueueUrl=_url(), ReceiptHandle=msg.receipt_handle)
