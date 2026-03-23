"""
Higgs Boson Detection - Model Training & Submission
Run: python train.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from xgboost import XGBClassifier


def main():
    # Load data
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    feature_cols = [c for c in train.columns if c != "label"]
    X = train[feature_cols].values
    y = train["label"].values.astype(int)
    X_test = test[feature_cols].values

    print(f"Train: {X.shape}, Test: {X_test.shape}")
    print(f"Label distribution: {np.bincount(y)}")

    # Cross-validation with XGBoost
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    scores = []

    xgb_params = {
        "n_estimators": 1000,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "auc",
        "early_stopping_rounds": 50,
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = XGBClassifier(**xgb_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=100,
        )

        val_proba = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_proba
        test_preds += model.predict_proba(X_test)[:, 1] / n_folds

        auc = roc_auc_score(y_val, val_proba)
        acc = accuracy_score(y_val, (val_proba > 0.5).astype(int))
        scores.append(auc)
        print(f"Fold {fold + 1} AUC: {auc:.5f}, Accuracy: {acc:.5f}")

    # Overall OOF score
    oof_auc = roc_auc_score(y, oof_preds)
    oof_acc = accuracy_score(y, (oof_preds > 0.5).astype(int))
    print(f"\n{'=' * 50}")
    print(f"CV AUC:  {np.mean(scores):.5f} (+/- {np.std(scores):.5f})")
    print(f"OOF AUC: {oof_auc:.5f}")
    print(f"OOF Acc: {oof_acc:.5f}")
    print(f"{'=' * 50}")

    # Feature importance
    print("\n--- Top 10 Features (last fold) ---")
    importance = model.feature_importances_
    feat_imp = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)
    for name, imp in feat_imp[:10]:
        print(f"  {name}: {imp:.4f}")

    # Generate submission
    submission = pd.read_csv("data/sample_submission.csv")
    submission["Predicted"] = (test_preds > 0.5).astype(int)
    submission.to_csv("submission.csv", index=False)
    print(f"\nSubmission saved: submission.csv")
    print(submission["Predicted"].value_counts())


if __name__ == "__main__":
    main()
