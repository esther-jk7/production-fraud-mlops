# Model Selection Rationale — Week 5

## Models Evaluated
| Model | Val PR-AUC | Test PR-AUC | Notes |
|-------|------------|-------------|-------|
| Logistic Regression (baseline) | 0.6807 | 0.7179 | class_weight=balanced |
| XGBoost (Optuna, 30 trials) | 0.8314 | 0.8648 | max_depth=3, lr=0.18 |
| LightGBM (Optuna, 30 trials) | 0.7682 | 0.8471 | 0.67ms inference latency |

## Winner: XGBoost

**Primary reason:** PR-AUC 0.8648 vs 0.8471 — XGBoost catches more fraud.

**Why PR-AUC over ROC-AUC?** With 577:1 class imbalance, ROC-AUC is misleading —
a model predicting all negatives scores ~0.5. PR-AUC focuses on the minority class
(fraud) and is the correct metric for this problem.

**Why not LightGBM despite faster inference?** In fraud detection, a missed fraud
(false negative) costs the bank the full transaction amount plus chargeback fees.
A 17ms vs 0.67ms latency difference is negligible in a payment pipeline.
PR-AUC improvement of +0.0177 is not negligible.

## Best XGBoost Hyperparameters
- n_estimators: 347
- max_depth: 3 (shallow trees generalize better on this dataset)
- learning_rate: 0.184
- subsample: 0.730
- colsample_bytree: 0.829
- min_child_weight: 2
- scale_pos_weight: 578.3 (handles class imbalance)

## Next Steps
- SHAP explanations on top 20 predictions
- Evidently AI drift monitoring
- Feature store with Feast

## Week 6 — Serving Layer

**FastAPI endpoints:**
- POST /predict — single transaction, 18ms latency
- POST /predict/batch — up to 1,000 transactions per call
- GET /stats — prediction statistics from SQLite log
- GET /health — model health check

**SQLite logging:** every prediction logged with timestamp, probability,
risk level, amount, and latency. Used for Week 7 drift detection.

**CI/CD:** ruff lint + pytest (14/14, 86% coverage) + Docker build
on every push to main.