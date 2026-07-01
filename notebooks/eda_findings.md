# EDA Findings — Credit Card Fraud Dataset

## Dataset
- 284,807 transactions, 31 features, 0 missing values
- Features V1-V28: PCA-transformed (anonymized) — no leakage risk
- Features requiring attention: Amount (raw, needs scaling), Time (raw, low signal)

## Class Imbalance
- Fraud: 492 (0.17%) | Legitimate: 284,315 (99.83%)
- Imbalance ratio: 577:1
- **Implication:** accuracy is meaningless. Use PR-AUC as primary metric.
- Will use stratified splits to preserve fraud ratio in train/val/test.

## Leakage Check
- V1-V28 are PCA components — no raw transaction data exposed
- Amount and Time are original features — no leakage
- No target leakage identified

## Top Features Correlated with Fraud
V17 (0.33), V14 (0.30), V12 (0.26), V10 (0.22), V16 (0.20)
These will be priority features to watch in SHAP analysis.

## Amount Distribution
- Mean: $88, Median: $22, Max: $25,691
- Fraud transactions tend to be smaller amounts (common pattern in card fraud)
- Needs StandardScaler before logistic regression