import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier


TARGET = "survived_to_18jan"
ID_COL = "apartment_id"


def build_pipeline(cat_cols, num_cols, random_state=42):
    # Числа: медиана + passthrough
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    # Категории: самый частый + onehot
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    # Бустинг по гистограммам: быстрый, устойчивый, хорошо для ROC-AUC
    clf = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=6,
        max_leaf_nodes=63,
        min_samples_leaf=20,
        l2_regularization=1e-4,
        max_bins=255,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=random_state,
    )

    return Pipeline([
        ("pre", pre),
        ("clf", clf),
    ])


def infer_column_types(df: pd.DataFrame):
    # категориальные: object + явно бинарные 0/1 можно оставить числом
    # но если они приходят строками — попадут в cat.
    cat_cols = [c for c in df.columns if df[c].dtype == "object"]
    num_cols = [c for c in df.columns if c not in cat_cols]
    return cat_cols, num_cols


def run_cv(train_df: pd.DataFrame, features: list, seed: int = 42, folds: int = 5):
    X = train_df[features].copy()
    y = train_df[TARGET].astype(int).values

    cat_cols, num_cols = infer_column_types(X)

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        model = build_pipeline(cat_cols, num_cols, random_state=seed + fold)
        model.fit(X_tr, y_tr)

        proba = model.predict_proba(X_va)[:, 1]
        score = roc_auc_score(y_va, proba)
        scores.append(score)
        print(f"Fold {fold}/{folds}: ROC-AUC = {score:.5f}")

    print(f"\nCV ROC-AUC: mean={float(np.mean(scores)):.5f} std={float(np.std(scores)):.5f}")
    return float(np.mean(scores))


def train_full_and_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, features: list, seed: int = 42):
    X_train = train_df[features].copy()
    y_train = train_df[TARGET].astype(int).values
    X_test = test_df[features].copy()

    cat_cols, num_cols = infer_column_types(pd.concat([X_train, X_test], axis=0, ignore_index=True))

    model = build_pipeline(cat_cols, num_cols, random_state=seed)
    model.fit(X_train, y_train)
    proba_test = model.predict_proba(X_test)[:, 1]
    proba_test = np.clip(proba_test, 0.0, 1.0)
    return proba_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--sample_sub", type=str, required=True)
    parser.add_argument("--out_csv", type=str, default="submission.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--cv", action="store_true", help="посчитать CV ROC-AUC перед финальным обучением")
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)
    sample = pd.read_csv(args.sample_sub)

    if ID_COL not in train_df.columns or ID_COL not in test_df.columns:
        raise ValueError(f"Не найден столбец {ID_COL} в train/test")

    if TARGET not in train_df.columns:
        raise ValueError(f"Не найден столбец цели {TARGET} в train")

    # список признаков: всё кроме id и target
    features = [c for c in train_df.columns if c not in [ID_COL, TARGET]]

    # простая санити-проверка: в test должны быть те же фичи
    missing_in_test = [c for c in features if c not in test_df.columns]
    if missing_in_test:
        raise ValueError(f"В test.csv нет колонок: {missing_in_test}")

    if args.cv:
        run_cv(train_df, features, seed=args.seed, folds=args.folds)

    proba_test = train_full_and_predict(train_df, test_df, features, seed=args.seed)

    sub = sample[[ID_COL]].merge(
        test_df[[ID_COL]].assign(survived_to_18jan=proba_test),
        on=ID_COL,
        how="left"
    )
    # если вдруг какого-то id нет — будет NaN, подставим 0.0
    sub["survived_to_18jan"] = sub["survived_to_18jan"].fillna(0.0).clip(0.0, 1.0)

    sub.to_csv(args.out_csv, index=False)
    print(f"Saved submission: {args.out_csv}")
    print(sub.head(10))


if __name__ == "__main__":
    main()
