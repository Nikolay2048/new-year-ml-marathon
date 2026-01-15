from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .features import ID_COL, TARGET_COL, add_features, get_feature_columns
from .target_encoding import add_oof_target_encoding
from .utils import Paths, ensure_dir, pretty_params, seed_everything


def build_model_params(seed: int, use_gpu: bool, device: str) -> Dict:
    """
    GPU-safe params (no rsm, no GPU-problematic ctr limits).
    Хорошая стартовая конфигурация под табличку.
    """
    params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": 5000,
        "learning_rate": 0.03,
        "depth": 8,
        "l2_leaf_reg": 6.0,
        "random_strength": 1.0,

        "bootstrap_type": "Bayesian",
        "bagging_temperature": 1.0,

        "od_type": "Iter",
        "od_wait": 200,
        "random_seed": seed,
        "verbose": 200,
        "allow_writing_files": False,
    }

    if use_gpu:
        params["task_type"] = "GPU"
        params["devices"] = device  # e.g. "0" or "0:1"
    else:
        params["task_type"] = "CPU"

    return params


def prepare_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    folds: int,
    seed: int,
    te_smoothing: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    # FE
    train_df = add_features(train_df)
    test_df = add_features(test_df)

    # base columns
    features, cat_cols, _num_cols = get_feature_columns(train_df)

    # fill categorical NaNs as string token
    for c in cat_cols:
        train_df[c] = train_df[c].astype("object").fillna("NA")
        test_df[c] = test_df[c].astype("object").fillna("NA")

    # OOF Target Encoding (powerful)
    train_df, test_df, te_feats = add_oof_target_encoding(
        train_df=train_df,
        test_df=test_df,
        cat_cols=cat_cols,
        target_col=TARGET_COL,
        n_splits=folds,
        seed=seed,
        smoothing=te_smoothing,
    )

    features = features + te_feats
    return train_df, test_df, features, cat_cols


def cv_train_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
    cat_cols: List[str],
    seed: int,
    folds: int,
    use_gpu: bool,
    device: str,
    outputs_dir: str,
) -> Tuple[np.ndarray, np.ndarray, float, List[float]]:
    y = train_df[TARGET_COL].values.astype(int)
    X = train_df[features]
    X_test = test_df[features]

    # categorical indices for Pool: only original categorical columns should be cat-features
    cat_idxs = [features.index(c) for c in cat_cols if c in features]

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    oof = np.zeros(len(train_df), dtype=float)
    test_pred = np.zeros(len(test_df), dtype=float)
    fold_scores: List[float] = []

    params = build_model_params(seed=seed, use_gpu=use_gpu, device=device)
    print("\nCatBoost params:\n" + pretty_params(params) + "\n")

    # store per-fold test preds (for debugging/ensembling variations)
    fold_test_preds = []

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
        fold_test = model.predict_proba(test_pool)[:, 1]
        fold_test_preds.append(fold_test)
        test_pred += fold_test / folds

        model_path = os.path.join(outputs_dir, f"model_fold{fold}.cbm")
        model.save_model(model_path)

        print(f"[Fold {fold}] AUC = {fold_auc:.6f} | best_iter={model.get_best_iteration()} | saved={model_path}")

    oof_auc = roc_auc_score(y, oof)
    print(f"\nFINAL OFFLINE METRIC (OOF ROC-AUC): {oof_auc:.6f}")
    print(f"Fold AUCs: {[round(s, 6) for s in fold_scores]}")
    print(f"Mean AUC: {np.mean(fold_scores):.6f} ± {np.std(fold_scores):.6f}\n")

    # save fold preds
    fold_preds_path = os.path.join(outputs_dir, "fold_test_preds.npy")
    np.save(fold_preds_path, np.vstack(fold_test_preds))
    print(f"Saved fold test preds: {fold_preds_path}")

    return oof, test_pred, oof_auc, fold_scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/train.csv")
    parser.add_argument("--test_path", type=str, default="data/test.csv")
    parser.add_argument("--sample_submission_path", type=str, default="data/sample_submission.csv")
    parser.add_argument("--outputs_dir", type=str, default="outputs")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)

    parser.add_argument("--use_gpu", action="store_true", help="Use CatBoost GPU")
    parser.add_argument("--device", type=str, default="0", help='GPU device string, e.g. "0" or "0:1"')

    parser.add_argument("--te_smoothing", type=float, default=20.0, help="Target encoding smoothing strength")
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

    assert ID_COL in train_df.columns and ID_COL in test_df.columns, "apartment_id missing"
    assert TARGET_COL in train_df.columns, "target missing in train"
    assert ID_COL in sample_sub.columns, "sample_submission must contain apartment_id"

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape:  {test_df.shape}")

    train_df, test_df, features, cat_cols = prepare_data(
        train_df=train_df,
        test_df=test_df,
        folds=args.folds,
        seed=args.seed,
        te_smoothing=args.te_smoothing,
    )

    print(f"\nFeatures count: {len(features)}")
    print(f"Categorical: {cat_cols}")

    oof, test_pred_cv, oof_auc, fold_scores = cv_train_predict(
        train_df=train_df,
        test_df=test_df,
        features=features,
        cat_cols=cat_cols,
        seed=args.seed,
        folds=args.folds,
        use_gpu=args.use_gpu,
        device=args.device,
        outputs_dir=paths.outputs_dir,
    )

    # save OOF
    oof_path = os.path.join(paths.outputs_dir, "oof.csv")
    pd.DataFrame(
        {ID_COL: train_df[ID_COL].values, "oof_pred": oof, TARGET_COL: train_df[TARGET_COL].values}
    ).to_csv(oof_path, index=False)
    print(f"Saved OOF: {oof_path}")

    # Build submission from CV ensemble (best practice)
    submission = pd.DataFrame(
        {
            ID_COL: test_df[ID_COL].values,
            TARGET_COL: np.clip(test_pred_cv, 0.0, 1.0),
        }
    )

    # reorder by sample_submission if needed
    if set(sample_sub[ID_COL].astype(str)) == set(submission[ID_COL].astype(str)):
        submission = sample_sub[[ID_COL]].merge(submission, on=ID_COL, how="left")

    sub_path = os.path.join(paths.outputs_dir, "submission.csv")
    submission.to_csv(sub_path, index=False)
    print(f"Saved submission: {sub_path}")

    print("\nSubmission preview:")
    print(submission.head())

    print("\nPred stats:")
    print(pd.Series(submission[TARGET_COL]).describe())


if __name__ == "__main__":
    main()
