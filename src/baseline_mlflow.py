import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, confusion_matrix
)
import matplotlib.pyplot as plt

# ── MLflow setup ──────────────────────────────────────────────────────────────
# Point MLflow at our local tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Create an experiment to group all fraud detection runs together
mlflow.set_experiment("fraud-detection")

# ── Load and split data ───────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("data/creditcard.csv")
X = df.drop('Class', axis=1)
y = df['Class']

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ── Train and log with MLflow ─────────────────────────────────────────────────
# Everything inside "with mlflow.start_run()" gets tracked automatically
with mlflow.start_run(run_name="baseline-logistic-regression"):

    # Log parameters — what settings did we use?
    params = {
        "model": "LogisticRegression",
        "class_weight": "balanced",
        "max_iter": 1000,
        "random_state": 42,
        "scaler": "StandardScaler",
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
    }
    mlflow.log_params(params)

    # Train
    print("Training...")
    model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate on validation
    y_val_pred = model.predict(X_val_scaled)
    y_val_prob = model.predict_proba(X_val_scaled)[:, 1]

    val_metrics = {
        "val_precision": precision_score(y_val, y_val_pred),
        "val_recall": recall_score(y_val, y_val_pred),
        "val_f1": f1_score(y_val, y_val_pred),
        "val_pr_auc": average_precision_score(y_val, y_val_prob),
    }

    # Evaluate on test
    y_test_pred = model.predict(X_test_scaled)
    y_test_prob = model.predict_proba(X_test_scaled)[:, 1]

    test_metrics = {
        "test_precision": precision_score(y_test, y_test_pred),
        "test_recall": recall_score(y_test, y_test_pred),
        "test_f1": f1_score(y_test, y_test_pred),
        "test_pr_auc": average_precision_score(y_test, y_test_prob),
    }

    # Log all metrics to MLflow
    mlflow.log_metrics({**val_metrics, **test_metrics})

    # Save confusion matrix as artifact
    cm = confusion_matrix(y_test, y_test_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Legitimate', 'Fraud'])
    ax.set_yticklabels(['Legitimate', 'Fraud'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix — Baseline Logistic Regression')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=14)
    plt.tight_layout()
    plt.savefig("confusion_matrix_baseline.png", dpi=150)
    mlflow.log_artifact("confusion_matrix_baseline.png")
    plt.close()

    # Log the model itself
    mlflow.sklearn.log_model(model, "model")

    print(f"\nRun logged to MLflow!")
    print(f"Val  PR-AUC: {val_metrics['val_pr_auc']:.4f}")
    print(f"Test PR-AUC: {test_metrics['test_pr_auc']:.4f}")
    print("\nOpen http://127.0.0.1:5000 to see the run.")