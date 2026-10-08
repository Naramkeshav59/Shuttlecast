# Build from the repo root:  docker build -f worker/Dockerfile -t shuttlecast-worker .
FROM python:3.11-slim

# nodejs: yt-dlp needs a JS runtime for YouTube's signature cipher. Only the
# YouTube fallback uses it; S3 is the primary video source.
RUN apt-get update \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# deps first so code edits don't invalidate the pip layers.
# CPU-only torch: Fargate has no GPU, and the CUDA wheels are ~2 GB bigger.
RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
COPY worker/requirements.txt worker/requirements.txt
RUN pip install --no-cache-dir -r worker/requirements.txt transformers==5.19.0

# Bake the frozen DINOv2 backbone into the image so a video job never waits
# on (or fails because of) a Hugging Face download at runtime.
ENV HF_HOME=/app/hf-cache
RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('facebook/dinov2-small')"

COPY shared/ shared/
COPY worker/ worker/
COPY vision/ vision/
COPY models/stroke_models.pt models/stroke_models.pt
COPY data/shuttleset/ data/shuttleset/
COPY data/verified_videos.json data/verified_videos.json

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HF_HUB_OFFLINE=1

# frames are written under data/videos/ at runtime
RUN useradd --create-home app && mkdir -p data/videos && chown -R app /app/data
USER app

CMD ["python", "-m", "worker.main"]
