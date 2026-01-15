from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def add_oof_target_encoding(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cat_cols: List[str],
    target_col: str,
    n_splits: int,
    seed: int,
    smoothing: float = 20.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    OOF Target Encoding without leakage.
    For each categorical column c:
      - For each fold: compute smoothed mean(target) per category on train-part
      - Apply to val-part (OOF)
      - Apply to test and average across folds
    """
    y = train_df[target_col].values.astype(float)
    global_mean = float(np.mean(y))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    train_out = train_df.copy()
    test_out = test_df.copy()
    new_feats: List[str] = []

    for c in cat_cols:
        feat = f"te_{c}"
        new_feats.append(feat)

        oof_vals = np.zeros(len(train_out), dtype=float)
        test_fold_vals = []

        for tr_idx, va_idx in skf.split(train_out, y.astype(int)):
            tr = train_out.iloc[tr_idx]
            va = train_out.iloc[va_idx]

            stats = tr.groupby(c)[target_col].agg(["mean", "count"])
            smooth = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)

            oof_vals[va_idx] = va[c].map(smooth).fillna(global_mean).values
            test_fold_vals.append(test_out[c].map(smooth).fillna(global_mean).values)

        train_out[feat] = oof_vals
        test_out[feat] = np.mean(np.vstack(test_fold_vals), axis=0)

    return train_out, test_out, new_feats
