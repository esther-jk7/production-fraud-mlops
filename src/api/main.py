import os
import pickle
import logging
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from xgboost import XGBClassifier

# ── Structured logging ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger(__name__)

# ── Global model state ────────────────────────────────────────────────────────
# We load the model once at startup and reuse it for all requests.
# Loading on every request would be 100x slower.
model = None
scaler = None
feature_names = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup, clean up at shutdown."""
    global model, scaler, feature_names

    logger.info("Loading model artifacts...")
    model = XGBClassifier()
    model.load_model("models/xgboost_fraud.json")

    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("models/feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)

    logger.info(f"Model loaded. Features: {len(feature_names)}")
    yield
    logger.info("Shutting down.")

app = FastAPI(
    title="Fraud Detection API",
    description="XGBoost fraud detection — PR-AUC 0.8648 on Credit Card Fraud dataset",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Pydantic schemas ──────────────────────────────────────────────────────────
# Why Pydantic? It validates every incoming request automatically.
# If someone sends a string where a float is expected, FastAPI returns
# a 422 error with a clear message — no manual validation code needed.

class TransactionFeatures(BaseModel):
    Time: float = Field(..., description="Seconds elapsed since first transaction")
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float = Field(..., ge=0, description="Transaction amount in dollars")

class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    risk_level: str  # LOW / MEDIUM / HIGH
    latency_ms: float

class HealthResponse(BaseModel):
    status: str
    model: str
    features: int

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthResponse(
        status="ok",
        model="xgboost-fraud-v1",
        features=len(feature_names)
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: TransactionFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()

    # Build feature array in correct order
    features = np.array([[getattr(transaction, f) for f in feature_names]])

    # Scale
    features_scaled = scaler.transform(features)

    # Predict
    fraud_prob = float(model.predict_proba(features_scaled)[0, 1])

    # Risk bucketing
    if fraud_prob < 0.3:
        risk_level = "LOW"
    elif fraud_prob < 0.7:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    latency_ms = (time.time() - start) * 1000
    logger.info(f"Prediction: prob={fraud_prob:.4f} risk={risk_level} latency={latency_ms:.2f}ms")

    return PredictionResponse(
        fraud_probability=round(fraud_prob, 6),
        is_fraud=fraud_prob >= 0.5,
        risk_level=risk_level,
        latency_ms=round(latency_ms, 2)
    )