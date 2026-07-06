import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from evidently import Dataset, DataDefinition
from evidently.presets import DataDriftPreset
from evidently import Report
import os

print("Loading data...")
df = pd.read_csv("data/creditcard.csv")
X = df.drop('Class', axis=1)
y = df['Class']

# ── Split exactly as during training ─────────────────────────────────────────
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

# ── Reference data = training set ────────────────────────────────────────────
reference_df = pd.DataFrame(X_train, columns=X.columns)
current_df = pd.DataFrame(X_test, columns=X.columns)

print(f"Reference data: {len(reference_df):,} rows")
print(f"Current data:   {len(current_df):,} rows")

# ── Generate drift report ─────────────────────────────────────────────────────
print("\nGenerating drift report...")

data_definition = DataDefinition()

reference_data = Dataset.from_pandas(
    reference_df,
    data_definition=data_definition
)

current_data = Dataset.from_pandas(
    current_df,
    data_definition=data_definition
)

report = Report(metrics=[DataDriftPreset()])
my_eval = report.run(reference_data=reference_data, current_data=current_data)

os.makedirs("reports", exist_ok=True)
my_eval.save_html("reports/drift_report.html")
print("Saved: reports/drift_report.html")
print("\nOpen reports/drift_report.html in your browser to see the full report.")