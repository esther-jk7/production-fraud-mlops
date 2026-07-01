import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("data/creditcard.csv")

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nDtypes:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())

# ── Class imbalance ───────────────────────────────────────────────────────────
class_counts = df['Class'].value_counts()
print("\nClass distribution:")
print(class_counts)
print(f"\nFraud rate: {class_counts[1] / len(df) * 100:.4f}%")
print(f"Imbalance ratio: {class_counts[0] / class_counts[1]:.1f}:1")

# ── Feature distributions ─────────────────────────────────────────────────────
print("\nAmount stats:")
print(df['Amount'].describe())

print("\nTime stats:")
print(df['Time'].describe())

# ── Leakage check ─────────────────────────────────────────────────────────────
print("\nCorrelation of features with Class (top 10):")
corr = df.corr()['Class'].abs().sort_values(ascending=False)
print(corr.head(11))

# ── Save plots ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].bar(['Legitimate', 'Fraud'], class_counts.values, color=['steelblue', 'crimson'])
axes[0].set_title('Class Distribution')
axes[0].set_ylabel('Count')
for i, v in enumerate(class_counts.values):
    axes[0].text(i, v + 100, f'{v:,}', ha='center')

axes[1].hist(df[df['Class']==0]['Amount'], bins=50, alpha=0.7, label='Legitimate', color='steelblue')
axes[1].hist(df[df['Class']==1]['Amount'], bins=50, alpha=0.7, label='Fraud', color='crimson')
axes[1].set_title('Transaction Amount Distribution')
axes[1].set_xlabel('Amount')
axes[1].set_ylabel('Count')
axes[1].legend()
axes[1].set_xlim(0, 2000)

plt.tight_layout()
plt.savefig('notebooks/eda_plots.png', dpi=150)
print("\nPlots saved to notebooks/eda_plots.png")