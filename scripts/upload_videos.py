"""Upload verified match clips to s3://<bucket>/videos/{match_id}.mp4.

The worker reads videos from S3 first because YouTube blocks datacenter IPs
far more aggressively than residential ones. Only
clips marked aligned in data/verified_videos.json are uploaded.

    python3 scripts/upload_videos.py [--stack shuttlecast]
"""
import argparse
import os
import sys
from pathlib import Path

import boto3

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from worker.infra import storage  # noqa: E402
from worker.pipeline.ingest import verified_videos  # noqa: E402


def bucket_from_stack(stack: str) -> str:
    outputs = boto3.client("cloudformation").describe_stacks(StackName=stack)["Stacks"][0]["Outputs"]
    return next(o["OutputValue"] for o in outputs if o["OutputKey"] == "BucketName")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack", default=os.environ.get("STACK_NAME", "shuttlecast"))
    args = parser.parse_args()

    os.environ.setdefault("S3_BUCKET_NAME", bucket_from_stack(args.stack))
    for match_id, path in verified_videos().items():
        key = storage.video_key(match_id)
        if storage.exists(key):
            print(f"skip (already in S3): {match_id}")
            continue
        print(f"uploading {path.name} ({path.stat().st_size / 1e6:.0f} MB) -> {key}")
        storage.upload_file(path, key)


if __name__ == "__main__":
    main()
