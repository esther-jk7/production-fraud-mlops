import pandas as pd
import numpy as np
from datetime import datetime, timezone

print("Creating Feast feature data...")
df = pd.read_csv("data/creditcard.csv")

# Feast requires an entity key and a timestamp column
df['transaction_id'] = range(len(df))
df['event_timestamp'] = datetime.now(timezone.utc)

# Save as parquet (Feast's preferred format)
feature_cols = ['transaction_id', 'event_timestamp'] + \
               [c for c in df.columns if c not in ['Class', 'transaction_id', 'event_timestamp', 'Time']]

df[feature_cols].to_parquet("data/feast_transactions.parquet", index=False)
print(f"Saved {len(df):,} transactions to data/feast_transactions.parquet")