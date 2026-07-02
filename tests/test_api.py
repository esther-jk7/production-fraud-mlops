import pickle
import pytest
import numpy as np
from fastapi.testclient import TestClient
from src.api.main import app, init_db
import src.api.main as api_module
from xgboost import XGBClassifier

# Load model before creating client
def _load_model():
    api_module.model = XGBClassifier()
    api_module.model.load_model("models/xgboost_fraud.json")
    with open("models/scaler.pkl", "rb") as f:
        api_module.scaler = pickle.load(f)
    with open("models/feature_names.pkl", "rb") as f:
        api_module.feature_names = pickle.load(f)
    init_db()

_load_model()
client = TestClient(app)

# ── Sample transactions ───────────────────────────────────────────────────────
FRAUD_TRANSACTION = {
    "Time": 406.0, "V1": -2.3122265423263, "V2": 1.95199201064158,
    "V3": -1.60985073229769, "V4": 3.9979055875468, "V5": -0.522187864667764,
    "V6": -1.42654531920595, "V7": -2.53738730624579, "V8": 1.39165724829804,
    "V9": -2.77008927719433, "V10": -2.77227214465915, "V11": 3.20203320709635,
    "V12": -2.89990738849473, "V13": -0.595221881324605, "V14": -4.28925378244217,
    "V15": 0.389724120274487, "V16": -1.14074717980657, "V17": -2.83005567450437,
    "V18": -0.0168224681808257, "V19": 0.416955705037907, "V20": 0.126910559061474,
    "V21": 0.517232370861764, "V22": -0.0350493686052974, "V23": -0.465211076182388,
    "V24": 0.320198198514526, "V25": 0.0445191674731724, "V26": 0.177839798284401,
    "V27": 0.261145002567677, "V28": -0.143275874698919, "Amount": 0.0
}

LEGIT_TRANSACTION = {
    "Time": 1.0, "V1": 1.19185711131486, "V2": 0.26615071205963,
    "V3": 0.16648011335321, "V4": 0.44815408489228, "V5": 0.06001765351328,
    "V6": -0.08236080663788, "V7": -0.07880298012891, "V8": 0.08510165259564,
    "V9": -0.25542063118613, "V10": -0.16697441396508, "V11": 1.61272666105479,
    "V12": 1.06523531137287, "V13": 0.48909501589608, "V14": -0.14377230692961,
    "V15": 0.63558505621094, "V16": 0.46391704127048, "V17": -0.11480466358788,
    "V18": -0.18336127639934, "V19": -0.14578304231496, "V20": -0.06908085809686,
    "V21": -0.22577522820609, "V22": -0.63867970235323, "V23": -0.03558429599378,
    "V24": 0.18136776025003, "V25": 0.51007738651603, "V26": -0.28792374956869,
    "V27": -0.14752266916158, "V28": -0.05671725547069, "Amount": 149.62
}

# ── Health tests ──────────────────────────────────────────────────────────────
def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_health_returns_model_name():
    response = client.get("/health")
    assert "model" in response.json()
    assert response.json()["model"] == "xgboost-fraud-v1"

def test_health_returns_feature_count():
    response = client.get("/health")
    assert response.json()["features"] == 30

# ── Predict tests ─────────────────────────────────────────────────────────────
def test_predict_fraud_transaction():
    response = client.post("/predict", json=FRAUD_TRANSACTION)
    assert response.status_code == 200
    data = response.json()
    assert data["is_fraud"] is True
    assert data["fraud_probability"] > 0.9
    assert data["risk_level"] == "HIGH"

def test_predict_legit_transaction():
    response = client.post("/predict", json=LEGIT_TRANSACTION)
    assert response.status_code == 200
    data = response.json()
    assert data["is_fraud"] is False
    assert data["fraud_probability"] < 0.1
    assert data["risk_level"] == "LOW"

def test_predict_response_schema():
    response = client.post("/predict", json=FRAUD_TRANSACTION)
    data = response.json()
    assert "fraud_probability" in data
    assert "is_fraud" in data
    assert "risk_level" in data
    assert "latency_ms" in data
    assert isinstance(data["fraud_probability"], float)
    assert isinstance(data["is_fraud"], bool)
    assert isinstance(data["latency_ms"], float)

def test_predict_invalid_missing_field():
    bad_request = {k: v for k, v in FRAUD_TRANSACTION.items() if k != "Amount"}
    response = client.post("/predict", json=bad_request)
    assert response.status_code == 422

def test_predict_invalid_negative_amount():
    bad_request = {**FRAUD_TRANSACTION, "Amount": -100}
    response = client.post("/predict", json=bad_request)
    assert response.status_code == 422

def test_predict_risk_levels():
    response = client.post("/predict", json=FRAUD_TRANSACTION)
    assert response.json()["risk_level"] in ["LOW", "MEDIUM", "HIGH"]

# ── Batch tests ───────────────────────────────────────────────────────────────
def test_batch_predict_two_transactions():
    response = client.post("/predict/batch", json={
        "transactions": [FRAUD_TRANSACTION, LEGIT_TRANSACTION]
    })
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert data["fraud_count"] == 1
    assert len(data["predictions"]) == 2

def test_batch_predict_empty_fails():
    response = client.post("/predict/batch", json={"transactions": []})
    assert response.status_code == 422

def test_batch_predict_response_schema():
    response = client.post("/predict/batch", json={
        "transactions": [LEGIT_TRANSACTION]
    })
    data = response.json()
    assert "predictions" in data
    assert "total" in data
    assert "fraud_count" in data
    assert "latency_ms" in data

# ── Stats tests ───────────────────────────────────────────────────────────────
def test_stats_returns_ok():
    response = client.get("/stats")
    assert response.status_code == 200

def test_stats_schema():
    response = client.get("/stats")
    data = response.json()
    assert "total_predictions" in data
    assert "fraud_count" in data
    assert "fraud_rate" in data
    assert "avg_fraud_probability" in data