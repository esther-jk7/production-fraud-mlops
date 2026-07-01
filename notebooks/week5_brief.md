# Week 5 Brief — Model Selection Rationale

## What We Built
End-to-end fraud detection baseline with production-class experiment tracking.

## Dataset
- 284,807 transactions, 492 fraud (0.17%), 577:1 imbalance ratio
- Features V1-V28 are PCA-transformed — no leakage risk
- Metric chosen: PR-AUC (not ROC-AUC) — correct for extreme imbalance

## Model Comparison
| Model | Test PR-AUC | Notes |
|-------|------------|-------|
| Logistic Regression | 0.7179 | Baseline, class_weight=balanced |
| XGBoost (Optuna, 30 trials) | 0.8648 | **Winner** |
| LightGBM (Optuna, 30 trials) | 0.8471 | Faster (0.67ms) but lower PR-AUC |

## Why XGBoost Won
PR-AUC 0.8648 vs 0.8471 — in fraud detection, catching more fraud outweighs
inference latency. A missed fraud costs the full transaction amount + chargeback
fees. 17ms vs 0.67ms latency difference is negligible in a payment pipeline.

## SHAP Findings
Top fraud-driving features by mean |SHAP value|:
1. V14 (2.51) — dominant predictor
2. V4  (2.43) — nearly as important
3. V12 (1.56), V3 (1.43), V10 (1.16)

Amount and Time do not appear in top 10 — PCA features carry all the signal.
This confirms no need for time-based feature engineering at this stage.

## MLflow Runs Logged
- baseline-logistic-regression
- xgboost-optuna-search (30 nested trial runs)
- lightgbm-optuna-search (30 nested trial runs)
- xgboost-shap-analysis (3 SHAP plots as artifacts)

## Next Week
- Evidently AI data drift monitoring
- Feast feature store
- FastAPI serving layer
- GitHub Actions CI/CD pipeline