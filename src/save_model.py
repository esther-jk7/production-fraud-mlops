import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("Loading data and training final model...")
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

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

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

# Save model and scaler
import os
os.makedirs("models", exist_ok=True)

model.save_model("models/xgboost_fraud.json")
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save feature names
feature_names = X.columns.tolist()
with open("models/feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)

print("Saved:")
print("  models/xgboost_fraud.json")
print("  models/scaler.pkl")
print("  models/feature_names.pkl")