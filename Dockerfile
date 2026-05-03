# syntax=docker/dockerfile:1.7

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install gcloud SDK to download model from GCS at build time
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates gnupg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
        | tee /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    apt-get update && apt-get install -y --no-install-recommends google-cloud-cli && \
    rm -rf /var/lib/apt/lists/*

COPY requirements-serving.txt .
RUN pip install -r requirements-serving.txt

COPY src/serving /app/src/serving

ARG MODEL_GCS_PATH
RUN test -n "$MODEL_GCS_PATH" || (echo "MODEL_GCS_PATH build arg required" && exit 1) && \
    gcloud storage cp -r "$MODEL_GCS_PATH" /app/models/banking77-distilbert

# Strip gcloud SDK after download to keep runtime image small
RUN apt-get remove -y --purge google-cloud-cli curl gnupg && \
    apt-get autoremove -y --purge && \
    rm -rf /var/lib/apt/lists/* /root/.config/gcloud /usr/lib/google-cloud-sdk

ENV MODEL_DIR=/app/models/banking77-distilbert \
    PORT=8080

EXPOSE 8080
CMD ["sh", "-c", "uvicorn src.serving.app:app --host 0.0.0.0 --port ${PORT}"]
