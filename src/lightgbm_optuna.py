import pandas as pd
import numpy as np
import mlflow
import mlflow.lightgbm
import optuna
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, confusion_matrix
)
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── MLflow setup ──────────────────────────────────────────────────────────────
mlflow.set_tracking_uri("http://127.0.0.1:5000")
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

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight: {scale_pos_weight:.1f}")

# ── Optuna objective ──────────────────────────────────────────────────────────
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "scale_pos_weight": scale_pos_weight,
        "random_state": 42,
        "verbosity": -1,
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_val_scaled, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
    )

    y_val_prob = model.predict_proba(X_val_scaled)[:, 1]
    pr_auc = average_precision_score(y_val, y_val_prob)

    with mlflow.start_run(run_name=f"lgbm-trial-{trial.number}", nested=True):
        mlflow.log_params(params)
        mlflow.log_metric("val_pr_auc", pr_auc)

    return pr_auc

# ── Run Optuna ────────────────────────────────────────────────────────────────
print("\nStarting Optuna search for LightGBM (30 trials)...")

with mlflow.start_run(run_name="lightgbm-optuna-search"):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30, show_progress_bar=True)

    best_params = study.best_params
    best_params["scale_pos_weight"] = scale_pos_weight
    best_params["random_state"] = 42
    best_params["verbosity"] = -1

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best val PR-AUC: {study.best_value:.4f}")

    # ── Train final model ─────────────────────────────────────────────────────
    print("\nTraining final LightGBM with best params...")
    final_model = lgb.LGBMClassifier(**best_params)
    final_model.fit(X_train_scaled, y_train, callbacks=[lgb.log_evaluation(-1)])

    # Evaluate
    y_val_pred = final_model.predict(X_val_scaled)
    y_val_prob = final_model.predict_proba(X_val_scaled)[:, 1]
    y_test_pred = final_model.predict(X_test_scaled)
    y_test_prob = final_model.predict_proba(X_test_scaled)[:, 1]

    # Measure inference latency
    start = time.time()
    for _ in range(100):
        final_model.predict_proba(X_test_scaled[:100])
    latency_ms = (time.time() - start) / 100 * 1000
    print(f"Inference latency: {latency_ms:.2f}ms per 100 transactions")

    val_metrics = {
        "val_precision": precision_score(y_val, y_val_pred),
        "val_recall": recall_score(y_val, y_val_pred),
        "val_f1": f1_score(y_val, y_val_pred),
        "val_pr_auc": average_precision_score(y_val, y_val_prob),
    }
    test_metrics = {
        "test_precision": precision_score(y_test, y_test_pred),
        "test_recall": recall_score(y_test, y_test_pred),
        "test_f1": f1_score(y_test, y_test_pred),
        "test_pr_auc": average_precision_score(y_test, y_test_prob),
        "inference_latency_ms": latency_ms,
    }

    mlflow.log_params(best_params)
    mlflow.log_metrics({**val_metrics, **test_metrics})

    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Legitimate', 'Fraud'])
    ax.set_yticklabels(['Legitimate', 'Fraud'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix — LightGBM (Best)')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=14)
    plt.tight_layout()
    plt.savefig("confusion_matrix_lgbm.png", dpi=150)
    mlflow.log_artifact("confusion_matrix_lgbm.png")
    plt.close()

    mlflow.lightgbm.log_model(final_model, "model")

    print(f"\n── LightGBM Final Results ──")
    print(f"Val  PR-AUC: {val_metrics['val_pr_auc']:.4f}")
    print(f"Test PR-AUC: {test_metrics['test_pr_auc']:.4f}")
    print(f"Latency:     {latency_ms:.2f}ms per 100 transactions")
    print(f"\n── Comparison ──")
    print(f"Baseline LR:  0.7179")
    print(f"XGBoost:      0.8648")
    print(f"LightGBM:     {test_metrics['test_pr_auc']:.4f}")