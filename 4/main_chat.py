import os
import re
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


# -----------------------
# Text cleaning
# -----------------------
_re_space = re.compile(r"\s+")
_re_url = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_re_user = re.compile(r"@\w+")
_re_num = re.compile(r"\d+")

def clean_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\u00A0", " ")
    s = s.replace("ё", "е").replace("Ё", "Е")
    s = s.lower()
    s = _re_url.sub(" URL ", s)
    s = _re_user.sub(" USER ", s)
    # числа часто не несут темы, но иногда важны (счета/доставка). Оставим маркер:
    s = _re_num.sub(" NUM ", s)
    s = _re_space.sub(" ", s).strip()
    return s


def make_pipeline(random_state: int = 42) -> Pipeline:
    """
    Pipeline:
    - text -> TF-IDF (word ngrams 1-2) + TF-IDF (char ngrams 3-5)
    - day/hour -> OneHot
    - classifier -> LinearSVC (хорошо для macro-F1)
    """

    # отдельные ветки: word tfidf + char tfidf
    word_tfidf = TfidfVectorizer(
        preprocessor=clean_text,
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    char_tfidf = TfidfVectorizer(
        preprocessor=clean_text,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
    )

    # ColumnTransformer не умеет напрямую смешивать два TF-IDF на одной колонке
    # поэтому делаем трюк: создаем две "копии" текста через FunctionTransformer
    # и подаем их в разные TF-IDF.
    def col(name):
        return FunctionTransformer(lambda X: X[name].astype(str).values, validate=False)

    # Конкатенация признаков:
    # - word tfidf (из text_word)
    # - char tfidf (из text_char)
    # - onehot day/hour
    pre = ColumnTransformer(
        transformers=[
            ("word", word_tfidf, "text_word"),
            ("char", char_tfidf, "text_char"),
            ("time", OneHotEncoder(handle_unknown="ignore"), ["day", "hour"]),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    clf = LinearSVC(
        C=1.0,
        class_weight="balanced",  # помогает macro-F1 при дисбалансе
        random_state=random_state,
    )

    pipe = Pipeline([
        ("pre", pre),
        ("clf", clf),
    ])
    return pipe


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # создаем 2 "копии" текста под word/char ветки
    df["text"] = df["text"].fillna("")
    df["text_word"] = df["text"]
    df["text_char"] = df["text"]
    # day/hour привести к int (на всякий)
    df["day"] = df["day"].astype(int)
    df["hour"] = df["hour"].astype(int)
    return df


def cv_train(train_df: pd.DataFrame, n_splits: int = 5, seed: int = 42):
    X = train_df[["text_word", "text_char", "day", "hour"]]
    y = train_df["topic"].astype(str).values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    scores = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        model = make_pipeline(random_state=seed)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_va)

        score = f1_score(y_va, pred, average="macro")
        scores.append(score)
        print(f"Fold {fold}/{n_splits}: macro-F1 = {score:.5f}")

    print(f"\nCV macro-F1: mean={np.mean(scores):.5f} std={np.std(scores):.5f}")
    return float(np.mean(scores))


def train_full_and_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, seed: int = 42):
    X_train = train_df[["text_word", "text_char", "day", "hour"]]
    y_train = train_df["topic"].astype(str).values

    X_test = test_df[["text_word", "text_char", "day", "hour"]]

    model = make_pipeline(random_state=seed)
    model.fit(X_train, y_train)
    test_pred = model.predict(X_test)

    return test_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True, help="train.csv (message_id,text,day,hour,topic)")
    parser.add_argument("--test_csv", type=str, required=True, help="test.csv (message_id,text,day,hour)")
    parser.add_argument("--sample_sub", type=str, required=True, help="sample_submission.csv (message_id,topic)")
    parser.add_argument("--out_csv", type=str, default="submission.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv", action="store_true", help="сделать 5-fold CV и вывести macro-F1")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)
    sample = pd.read_csv(args.sample_sub)

    train_df = prepare_df(train_df)
    test_df = prepare_df(test_df)

    if args.cv:
        cv_train(train_df, n_splits=args.folds, seed=args.seed)

    test_pred = train_full_and_predict(train_df, test_df, seed=args.seed)

    # Собираем submission в порядке sample_submission (если вдруг порядок важен)
    sub = sample[["message_id"]].merge(
        test_df[["message_id"]].assign(topic=test_pred),
        on="message_id",
        how="left"
    )
    # если вдруг чего-то нет — ставим "other_chat"
    sub["topic"] = sub["topic"].fillna("other_chat")

    sub.to_csv(args.out_csv, index=False)
    print(f"Saved: {args.out_csv}")
    print(sub.head(10))


if __name__ == "__main__":
    main()
