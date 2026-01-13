from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .features import TARGET_COL, ID_COL, add_features, get_feature_columns
from .utils import Paths, ensure_dir, pretty_params, seed_everything


def build_model_params(seed: int) -> Dict:
    return {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": 5000,
        "learning_rate": 0.03,
        "depth": 8,
        "l2_leaf_reg": 6.0,
        "random_seed": seed,
        "od_type": "Iter",
        "od_wait": 200,
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 1.0,  # <-- вместо subsample
        "random_strength": 1.0,
        "verbose": 200,
        "allow_writing_files": False,
    }


def prepare_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    # Feature engineering
    train_df = add_features(train_df)
    test_df = add_features(test_df)

    # Identify columns
    features, cat_cols, _num_cols = get_feature_columns(train_df)

    # Fill categorical NaNs (CatBoost expects strings for cat features)
    for c in cat_cols:
        train_df[c] = train_df[c].astype("object").fillna("NA")
        test_df[c] = test_df[c].astype("object").fillna("NA")

    return train_df, test_df, features, cat_cols


def cv_train_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
    cat_cols: List[str],
    seed: int,
    n_splits: int,
) -> Tuple[np.ndarray, np.ndarray, float, List[float]]:
    y = train_df[TARGET_COL].values.astype(int)
    X = train_df[features]
    X_test = test_df[features]

    cat_idxs = [features.index(c) for c in cat_cols]  # indices for Pool

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof = np.zeros(len(train_df), dtype=float)
    test_pred = np.zeros(len(test_df), dtype=float)
    fold_scores: List[float] = []

    params = build_model_params(seed)
    print("\nCatBoost params:\n" + pretty_params(params) + "\n")

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        train_pool = Pool(X_tr, y_tr, cat_features=cat_idxs)
        valid_pool = Pool(X_va, y_va, cat_features=cat_idxs)

        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

        va_pred = model.predict_proba(valid_pool)[:, 1]
        oof[va_idx] = va_pred
        fold_auc = roc_auc_score(y_va, va_pred)
        fold_scores.append(fold_auc)

        test_pool = Pool(X_test, cat_features=cat_idxs)
        test_pred += model.predict_proba(test_pool)[:, 1] / n_splits

        print(f"[Fold {fold}] AUC = {fold_auc:.6f} | best_iter={model.get_best_iteration()}")

    oof_auc = roc_auc_score(y, oof)
    print(f"\nOOF AUC = {oof_auc:.6f}")
    print(f"Fold AUCs: {[round(s, 6) for s in fold_scores]}")
    print(f"Mean AUC: {np.mean(fold_scores):.6f} ± {np.std(fold_scores):.6f}\n")

    return oof, test_pred, oof_auc, fold_scores


def train_full_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
    cat_cols: List[str],
    seed: int,
) -> Tuple[CatBoostClassifier, np.ndarray]:
    y = train_df[TARGET_COL].values.astype(int)
    X = train_df[features]
    X_test = test_df[features]

    cat_idxs = [features.index(c) for c in cat_cols]

    params = build_model_params(seed)
    # На full-train можно чуть больше итераций, но оставим те же: od отработает.
    model = CatBoostClassifier(**params)

    train_pool = Pool(X, y, cat_features=cat_idxs)
    model.fit(train_pool)

    test_pool = Pool(X_test, cat_features=cat_idxs)
    test_pred = model.predict_proba(test_pool)[:, 1]
    return model, test_pred


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/train.csv")
    parser.add_argument("--test_path", type=str, default="data/test.csv")
    parser.add_argument("--sample_submission_path", type=str, default="data/sample_submission.csv")
    parser.add_argument("--outputs_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    seed_everything(args.seed)

    paths = Paths(
        train_path=args.train_path,
        test_path=args.test_path,
        sample_submission_path=args.sample_submission_path,
        outputs_dir=args.outputs_dir,
    )
    ensure_dir(paths.outputs_dir)

    print("Loading data...")
    train_df = pd.read_csv(paths.train_path)
    test_df = pd.read_csv(paths.test_path)
    sample_sub = pd.read_csv(paths.sample_submission_path)

    # Basic checks
    assert ID_COL in train_df.columns and ID_COL in test_df.columns, "apartment_id missing"
    assert TARGET_COL in train_df.columns, "target missing in train"
    assert ID_COL in sample_sub.columns, "sample submission must contain apartment_id"

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape:  {test_df.shape}")

    train_df, test_df, features, cat_cols = prepare_data(train_df, test_df)

    print(f"\nFeatures count: {len(features)}")
    print(f"Categorical: {cat_cols}")

    # CV training (for confidence)
    oof, test_pred_cv, oof_auc, fold_scores = cv_train_predict(
        train_df=train_df,
        test_df=test_df,
        features=features,
        cat_cols=cat_cols,
        seed=args.seed,
        n_splits=args.folds,
    )

    # Save OOF for later analysis
    oof_path = os.path.join(paths.outputs_dir, "oof.csv")
    pd.DataFrame({ID_COL: train_df[ID_COL].values, "oof_pred": oof, TARGET_COL: train_df[TARGET_COL].values}).to_csv(
        oof_path, index=False
    )
    print(f"Saved OOF: {oof_path}")

    # Train final model on full data
    print("\nTraining full model on all train data...")
    model, test_pred = train_full_model(
        train_df=train_df,
        test_df=test_df,
        features=features,
        cat_cols=cat_cols,
        seed=args.seed,
    )

    model_path = os.path.join(paths.outputs_dir, "model.cbm")
    model.save_model(model_path)
    print(f"Saved model: {model_path}")

    # Build submission
    submission = pd.DataFrame(
        {
            ID_COL: test_df[ID_COL].values,
            TARGET_COL: np.clip(test_pred, 0.0, 1.0),
        }
    )

    # Optional: keep original ordering as in sample_submission (if needed)
    # If sample_submission has exact same apartment_id set, we can reorder:
    if set(sample_sub[ID_COL].astype(str)) == set(submission[ID_COL].astype(str)):
        submission = sample_sub[[ID_COL]].merge(submission, on=ID_COL, how="left")

    sub_path = os.path.join(paths.outputs_dir, "submission.csv")
    submission.to_csv(sub_path, index=False)
    print(f"Saved submission: {sub_path}")

    # Print quick sanity stats
    print("\nSubmission preview:")
    print(submission.head())
    print("\nPred stats:")
    print(pd.Series(submission[TARGET_COL]).describe())

