import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
import shap
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

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

# Convert back to DataFrame for SHAP (needs feature names)
X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# ── Train best XGBoost model ──────────────────────────────────────────────────
# These are the best params from the Optuna search
print("Training XGBoost with best params...")
best_params = {
    "n_estimators": 347,
    "max_depth": 3,
    "learning_rate": 0.18461613004105523,
    "subsample": 0.7303376779229789,
    "colsample_bytree": 0.8290397563747758,
    "min_child_weight": 2,
    "scale_pos_weight": scale_pos_weight,
    "random_state": 42,
    "eval_metric": "aucpr",
}

model = XGBClassifier(**best_params)
model.fit(X_train_scaled, y_train, verbose=False)

y_test_prob = model.predict_proba(X_test_scaled)[:, 1]
print(f"Test PR-AUC: {average_precision_score(y_test, y_test_prob):.4f}")

# ── SHAP Analysis ─────────────────────────────────────────────────────────────
print("\nComputing SHAP values (this takes ~1 minute)...")

# TreeExplainer is optimized for tree-based models like XGBoost
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_df)

print(f"SHAP values shape: {shap_values.shape}")

with mlflow.start_run(run_name="xgboost-shap-analysis"):
    mlflow.log_params(best_params)
    mlflow.log_metric("test_pr_auc", average_precision_score(y_test, y_test_prob))

    # ── Plot 1: Global feature importance (summary plot) ──────────────────────
    # Shows which features matter most ACROSS ALL predictions
    # Each dot = one transaction. Color = feature value (red=high, blue=low)
    # X-axis = SHAP value (impact on fraud probability)
    print("Generating summary plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_df, show=False, max_display=20)
    plt.title("SHAP Summary — Feature Impact on Fraud Probability", fontsize=13)
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    mlflow.log_artifact("shap_summary.png")
    print("Saved shap_summary.png")

    # ── Plot 2: Mean absolute SHAP values (bar chart) ─────────────────────────
    # Simpler view — just the average importance of each feature
    print("Generating bar plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_df, plot_type="bar", show=False, max_display=20)
    plt.title("Mean |SHAP Value| — Average Feature Importance", fontsize=13)
    plt.tight_layout()
    plt.savefig("shap_bar.png", dpi=150, bbox_inches='tight')
    plt.close()
    mlflow.log_artifact("shap_bar.png")
    print("Saved shap_bar.png")

    # ── Plot 3: Top 20 fraud predictions — individual explanations ────────────
    # Find the top 20 transactions with highest fraud probability
    test_indices = np.where(y_test == 1)[0]  # actual fraud cases
    fraud_probs = y_test_prob[test_indices]
    top20_idx = test_indices[np.argsort(fraud_probs)[-20:]]

    print("Generating waterfall plots for top 20 fraud predictions...")
    fig, axes = plt.subplots(4, 5, figsize=(25, 20))
    axes = axes.flatten()

    for plot_idx, test_idx in enumerate(top20_idx):
        ax = axes[plot_idx]
        shap_vals = shap_values[test_idx]
        top_features = np.argsort(np.abs(shap_vals))[-5:]

        colors = ['crimson' if v > 0 else 'steelblue' for v in shap_vals[top_features]]
        ax.barh(
            X_test_df.columns[top_features],
            shap_vals[top_features],
            color=colors
        )
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.set_title(f"Fraud #{plot_idx+1} | P={fraud_probs[np.argsort(fraud_probs)[-20:][plot_idx]]:.3f}", fontsize=9)
        ax.tick_params(axis='y', labelsize=7)

    plt.suptitle("Top 20 Fraud Predictions — Top 5 SHAP Features Each", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("shap_top20_fraud.png", dpi=120, bbox_inches='tight')
    plt.close()
    mlflow.log_artifact("shap_top20_fraud.png")
    print("Saved shap_top20_fraud.png")

    # ── Print top features ────────────────────────────────────────────────────
    mean_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'mean_abs_shap': mean_shap
    }).sort_values('mean_abs_shap', ascending=False)

    print("\n── Top 10 Features by Mean |SHAP| ──")
    print(feature_importance.head(10).to_string(index=False))

    mlflow.log_artifact("shap_summary.png")

print("\nSHAP analysis complete. Check MLflow for plots.")