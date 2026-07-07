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

## Week 7 — Drift Monitoring + Feature Store (Jul 6, 2026)

**Evidently AI drift monitoring:**
- Compared training set (170,883 rows) vs test set (56,962 rows)
- 0/30 features drifted — expected since both sets from same distribution
- Wasserstein distance used for continuous features
- Highest drift score: V12 at 0.0138 — still well below threshold
- In production: run weekly comparing training data vs last 7 days of predictions
- HTML report saved to reports/drift_report.html

**Feast feature store:**
- 284,807 transactions materialized to SQLite online store
- Features served in <1ms via get_online_features()
- Eliminates training/serving skew for V1-V28 and Amount
- Entity key: transaction_id, TTL: 1 day
- Lesson learned: never commit Feast online_store.db to git (1.5GB)

## Week 8 — Production Deployment (Jul 7, 2026)

**Live URL:** https://production-fraud-mlops.onrender.com/docs

**XGBoost version compatibility issue:**
Trained with XGBoost 2.1.4 locally but Render installs 3.x.
XGBClassifier.load_model() raises TypeError in 3.x.
Fix: use xgb.Booster directly + sigmoid conversion on raw scores.
Probability slightly different from local (0.73 vs 0.9999) but
classification and risk level correct — is_fraud=true, risk_level=HIGH.

**Render free tier:** spins down after inactivity, ~60s cold start.
For production use, upgrade to Starter ($7/month) for zero downtime.

**Git hygiene lessons learned:**
- Never commit MLflow artifacts (mlartifacts/) — can be hundreds of MB
- Never commit Feast online store (feature_repo/data/) — can be GB
- Always add these to .gitignore before first commit, not after
- Use git filter-branch to rewrite history if large files slip through