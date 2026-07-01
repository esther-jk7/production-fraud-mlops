import os
import pickle
import logging
import time
import sqlite3
from datetime import datetime
from typing import List
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
    init_db()
    logger.info("Database initialized.")
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

# ── SQLite prediction logging ─────────────────────────────────────────────────
# Why SQLite? Lightweight, no server needed, perfect for a single-instance
# deployment. Week 7 drift detection will query this table.
DB_PATH = "predictions.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            fraud_probability REAL,
            is_fraud INTEGER,
            risk_level TEXT,
            amount REAL,
            latency_ms REAL
        )
    """)
    conn.commit()
    conn.close()

def log_prediction(fraud_prob: float, is_fraud: bool, risk_level: str,
                   amount: float, latency_ms: float):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO predictions
        (timestamp, fraud_probability, is_fraud, risk_level, amount, latency_ms)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (datetime.utcnow().isoformat(), fraud_prob, int(is_fraud),
          risk_level, amount, latency_ms))
    conn.commit()
    conn.close()

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
    log_prediction(fraud_prob, fraud_prob >= 0.5, risk_level,transaction.Amount, latency_ms)

    return PredictionResponse(
        fraud_probability=round(fraud_prob, 6),
        is_fraud=fraud_prob >= 0.5,
        risk_level=risk_level,
        latency_ms=round(latency_ms, 2)
    )
class BatchRequest(BaseModel):
    transactions: List[TransactionFeatures] = Field(
        ..., min_length=1, max_length=1000,
        description="List of transactions to score"
    )

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total: int
    fraud_count: int
    latency_ms: float

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()

    features = np.array([
        [getattr(t, f) for f in feature_names]
        for t in request.transactions
    ])

    features_scaled = scaler.transform(features)
    fraud_probs = model.predict_proba(features_scaled)[:, 1]

    predictions = []
    for i, (transaction, prob) in enumerate(zip(request.transactions, fraud_probs)):
        prob = float(prob)
        if prob < 0.3:
            risk_level = "LOW"
        elif prob < 0.7:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        log_prediction(prob, prob >= 0.5, risk_level, transaction.Amount, 0)
        predictions.append(PredictionResponse(
            fraud_probability=round(prob, 6),
            is_fraud=prob >= 0.5,
            risk_level=risk_level,
            latency_ms=0
        ))

    total_latency = (time.time() - start) * 1000
    fraud_count = sum(1 for p in predictions if p.is_fraud)

    logger.info(f"Batch: {len(predictions)} transactions, {fraud_count} fraud, {total_latency:.2f}ms")

    return BatchPredictionResponse(
        predictions=predictions,
        total=len(predictions),
        fraud_count=fraud_count,
        latency_ms=round(total_latency, 2)
    )

@app.get("/stats")
async def stats():
    """Return prediction statistics from the SQLite log."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("""
        SELECT
            COUNT(*) as total,
            SUM(is_fraud) as fraud_count,
            AVG(fraud_probability) as avg_prob,
            AVG(latency_ms) as avg_latency
        FROM predictions
    """)
    row = cursor.fetchone()
    conn.close()

    return {
        "total_predictions": row[0],
        "fraud_count": row[1],
        "fraud_rate": round(row[1] / row[0], 4) if row[0] > 0 else 0,
        "avg_fraud_probability": round(row[2], 4) if row[2] else 0,
        "avg_latency_ms": round(row[3], 2) if row[3] else 0,
    }