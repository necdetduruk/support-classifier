"""FastAPI inference service for banking77 intent classification."""
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# --- Config ---
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models/banking77-distilbert"))
MAX_LENGTH = 64
TOP_K_DEFAULT = 3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("serving")

# Loaded at startup, populated in lifespan
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once at startup."""
    logger.info(f"Loading model from {MODEL_DIR}")
    if not MODEL_DIR.exists():
        raise RuntimeError(f"Model directory not found: {MODEL_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.eval()

    with open(MODEL_DIR / "label_names.json") as f:
        label_names = json.load(f)

    metrics_path = MODEL_DIR / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

    state["tokenizer"] = tokenizer
    state["model"] = model
    state["label_names"] = label_names
    state["metrics"] = metrics
    logger.info(f"Model loaded. {len(label_names)} labels. Metrics: {metrics}")
    yield
    state.clear()


app = FastAPI(
    title="Banking77 Intent Classifier",
    version="0.1.0",
    lifespan=lifespan,
)


# --- Schemas ---
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=512)
    top_k: int = Field(default=TOP_K_DEFAULT, ge=1, le=10)


class Prediction(BaseModel):
    label: str
    score: float


class PredictResponse(BaseModel):
    text: str
    predictions: List[Prediction]
    model_version: str = "banking77-distilbert-v1"


# --- Routes ---
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/info")
def info():
    return {
        "model_version": "banking77-distilbert-v1",
        "num_labels": len(state.get("label_names", [])),
        "metrics": state.get("metrics", {}),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if "model" not in state:
        raise HTTPException(status_code=503, detail="Model not loaded")

    tokenizer = state["tokenizer"]
    model = state["model"]
    labels = state["label_names"]

    inputs = tokenizer(
        req.text,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

    top_k = min(req.top_k, len(labels))
    top_probs, top_idx = torch.topk(probs, k=top_k)

    predictions = [
        Prediction(label=labels[i.item()], score=float(p.item()))
        for p, i in zip(top_probs, top_idx)
    ]

    logger.info(
        f"prediction text_len={len(req.text)} top1={predictions[0].label} "
        f"top1_score={predictions[0].score:.4f}"
    )

    return PredictResponse(text=req.text, predictions=predictions)