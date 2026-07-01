import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("data/creditcard.csv")

X = df.drop('Class', axis=1)
y = df['Class']

print(f"Total samples: {len(df):,}")
print(f"Fraud cases: {y.sum():,} ({y.mean()*100:.3f}%)")

# ── Stratified split ──────────────────────────────────────────────────────────
# Why stratified? With 577:1 imbalance, random splits could put almost no
# fraud cases in val/test. Stratified guarantees fraud ratio is preserved.
print("\nSplitting data (60/20/20 stratified)...")
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

print(f"Train: {len(X_train):,} samples | Fraud: {y_train.sum()}")
print(f"Val:   {len(X_val):,} samples | Fraud: {y_val.sum()}")
print(f"Test:  {len(X_test):,} samples | Fraud: {y_test.sum()}")

# ── Scale features ────────────────────────────────────────────────────────────
# Why scale? Logistic regression is sensitive to feature magnitude.
# Amount ranges 0-25,691 while V1-V28 are already PCA-normalized (~-5 to 5).
# Without scaling, Amount dominates. We fit scaler on TRAIN only to prevent
# data leakage into val/test.
print("\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ── Train logistic regression ─────────────────────────────────────────────────
# class_weight='balanced' tells sklearn to weight fraud cases 577x higher
# so the model doesn't just predict "not fraud" for everything.
print("\nTraining logistic regression...")
model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
def evaluate(model, X, y, split_name, scaler=None):
    if scaler:
        X = scaler.transform(X)
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    pr_auc = average_precision_score(y, y_prob)

    print(f"\n── {split_name} ──")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred))

    return {"precision": precision, "recall": recall, "f1": f1, "pr_auc": pr_auc}

val_metrics = evaluate(model, X_val_scaled, y_val, "Validation")
test_metrics = evaluate(model, X_test_scaled, y_test, "Test")

print("\n── Baseline Summary ──")
print(f"Val  PR-AUC: {val_metrics['pr_auc']:.4f}")
print(f"Test PR-AUC: {test_metrics['pr_auc']:.4f}")
print("\nThis is our floor. XGBoost must beat this.")