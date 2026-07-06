import sys
sys.path.insert(0, '/Users/estherkeerthi/production-fraud-mlops/feature_repo/feature_repo')

from feast import FeatureStore
import pandas as pd

# Connect to the feature store
store = FeatureStore(repo_path="/Users/estherkeerthi/production-fraud-mlops/feature_repo/feature_repo")

# Fetch features for 3 transactions by their IDs
entity_df = pd.DataFrame({
    "transaction_id": [0, 1, 2],
})

features = store.get_online_features(
    features=[
        "transaction_features:V1",
        "transaction_features:V4",
        "transaction_features:V14",
        "transaction_features:Amount",
    ],
    entity_rows=entity_df.to_dict(orient="records")
).to_df()

print("Features retrieved from Feast online store:")
print(features)