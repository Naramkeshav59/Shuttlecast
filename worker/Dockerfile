# Build from the repo root:  docker build -f worker/Dockerfile -t shuttlecast-worker .
FROM python:3.11-slim

# nodejs: yt-dlp needs a JS runtime for YouTube's signature cipher. Only the
# YouTube fallback uses it; S3 is the primary video source.
RUN apt-get update \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# deps first so code edits don't invalidate the pip layer
COPY worker/requirements.txt worker/requirements.txt
RUN pip install --no-cache-dir -r worker/requirements.txt

COPY shared/ shared/
COPY worker/ worker/
COPY data/shuttleset/ data/shuttleset/
COPY data/verified_videos.json data/verified_videos.json

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# frames are written under data/videos/ at runtime
RUN useradd --create-home app && mkdir -p data/videos && chown -R app /app/data
USER app

CMD ["python", "-m", "worker.main"]
