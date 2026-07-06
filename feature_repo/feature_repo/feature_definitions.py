from datetime import timedelta
from feast import Entity, Feature, FeatureView, FileSource, ValueType, Field
from feast.types import Float64, Int64

# ── Data source ───────────────────────────────────────────────────────────────
# Feast reads features from a file source during training (offline store)
# and serves them from SQLite during inference (online store)
transaction_source = FileSource(
    path="/Users/estherkeerthi/production-fraud-mlops/data/feast_transactions.parquet",
    timestamp_field="event_timestamp",
)

# ── Entity ────────────────────────────────────────────────────────────────────
# An entity is the primary key — what you're making predictions about.
# In fraud detection, each transaction has a unique ID.
transaction = Entity(
    name="transaction_id",
    value_type=ValueType.INT64,
    description="Unique transaction identifier"
)

# ── Feature View ──────────────────────────────────────────────────────────────
# A FeatureView groups related features together with a TTL (time-to-live).
# TTL means: after this long, features are considered stale.
transaction_features = FeatureView(
    name="transaction_features",
    entities=[transaction],
    ttl=timedelta(days=1),
    schema=[
        Field(name="V1", dtype=Float64),
        Field(name="V2", dtype=Float64),
        Field(name="V3", dtype=Float64),
        Field(name="V4", dtype=Float64),
        Field(name="V5", dtype=Float64),
        Field(name="V6", dtype=Float64),
        Field(name="V7", dtype=Float64),
        Field(name="V8", dtype=Float64),
        Field(name="V9", dtype=Float64),
        Field(name="V10", dtype=Float64),
        Field(name="V11", dtype=Float64),
        Field(name="V12", dtype=Float64),
        Field(name="V13", dtype=Float64),
        Field(name="V14", dtype=Float64),
        Field(name="V15", dtype=Float64),
        Field(name="V16", dtype=Float64),
        Field(name="V17", dtype=Float64),
        Field(name="V18", dtype=Float64),
        Field(name="V19", dtype=Float64),
        Field(name="V20", dtype=Float64),
        Field(name="V21", dtype=Float64),
        Field(name="V22", dtype=Float64),
        Field(name="V23", dtype=Float64),
        Field(name="V24", dtype=Float64),
        Field(name="V25", dtype=Float64),
        Field(name="V26", dtype=Float64),
        Field(name="V27", dtype=Float64),
        Field(name="V28", dtype=Float64),
        Field(name="Amount", dtype=Float64),
    ],
    source=transaction_source,
)