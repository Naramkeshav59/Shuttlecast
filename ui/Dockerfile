# Build from the repo root:  docker build -f ui/Dockerfile -t shuttlecast-ui .
# Runs in AWS mode: set SHUTTLECAST_API_URL (the ECS task definition does).
FROM python:3.11-slim

WORKDIR /app

COPY ui/requirements.txt ui/requirements.txt
RUN pip install --no-cache-dir -r ui/requirements.txt

COPY shared/ shared/
COPY ui/ ui/
COPY data/verified_videos.json data/verified_videos.json
COPY data/shuttleset/set/match.csv data/shuttleset/set/match.csv

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

RUN useradd --create-home app
USER app

EXPOSE 8501
CMD ["streamlit", "run", "ui/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
