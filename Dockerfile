# syntax=docker/dockerfile:1.7

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements-serving.txt .
RUN pip install -r requirements-serving.txt

COPY src/serving /app/src/serving
COPY models/banking77-distilbert /app/models/banking77-distilbert

ENV MODEL_DIR=/app/models/banking77-distilbert \
    PORT=8080

EXPOSE 8080
CMD ["sh", "-c", "uvicorn src.serving.app:app --host 0.0.0.0 --port ${PORT}"]


