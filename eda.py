"""
Exploratory Data Analysis for Higgs Boson Detection dataset.
Run: python eda.py
"""

import pandas as pd
import numpy as np


def main():
    print("=" * 60)
    print("HIGGS BOSON DETECTION - EDA")
    print("=" * 60)

    # Load data
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")

    # Basic info
    print(f"\n--- Train shape: {train.shape} ---")
    print(f"--- Test shape:  {test.shape} ---")
    print(f"--- Submission shape: {sample_sub.shape} ---")

    # Columns
    print(f"\n--- Train columns ---")
    print(train.columns.tolist())

    print(f"\n--- Test columns ---")
    print(test.columns.tolist())

    print(f"\n--- Sample submission columns ---")
    print(sample_sub.columns.tolist())

    # First few rows
    print(f"\n--- Train head ---")
    print(train.head())

    print(f"\n--- Train dtypes ---")
    print(train.dtypes)

    # Target distribution
    # Try to find the label/target column
    possible_targets = ["Label", "label", "target", "Target", "class", "Class"]
    target_col = None
    for col in possible_targets:
        if col in train.columns:
            target_col = col
            break

    if target_col:
        print(f"\n--- Target column: '{target_col}' ---")
        print(train[target_col].value_counts())
        print(f"\nTarget distribution (%):")
        print(train[target_col].value_counts(normalize=True) * 100)
    else:
        print("\n--- Could not auto-detect target column ---")
        print("Last 3 columns:", train.columns[-3:].tolist())

    # Missing values
    print(f"\n--- Missing values (train) ---")
    missing = train.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("No missing values")

    # Check for -999.0 sentinel values (common in HEP datasets)
    sentinel_counts = (train == -999.0).sum()
    if sentinel_counts.sum() > 0:
        print(f"\n--- Columns with -999.0 sentinel values ---")
        print(sentinel_counts[sentinel_counts > 0])

    # Numeric summary
    print(f"\n--- Numeric summary ---")
    print(train.describe())

    # Weight column check
    weight_cols = [c for c in train.columns if "weight" in c.lower()]
    if weight_cols:
        print(f"\n--- Weight columns found: {weight_cols} ---")
        for wc in weight_cols:
            print(f"{wc}: min={train[wc].min():.4f}, max={train[wc].max():.4f}, mean={train[wc].mean():.4f}")

    print("\n" + "=" * 60)
    print("EDA complete. Use these insights to build the model.")
    print("=" * 60)


if __name__ == "__main__":
    main()
