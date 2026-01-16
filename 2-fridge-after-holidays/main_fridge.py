import os
import re
import math
import json
import time
import random
import argparse
from typing import Dict, List, Tuple, Iterable, Any

import numpy as np
import pandas as pd

from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction import FeatureHasher


# -----------------------
# Helpers: parsing tags
# -----------------------
def split_pipe(s: Any) -> List[str]:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return []
    s = str(s).strip()
    if not s:
        return []
    return [t.strip() for t in s.split("|") if t.strip()]


def safe_int(x, default=0):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return default
        return int(x)
    except Exception:
        return default


def safe_float(x, default=0.0):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return default
        return float(x)
    except Exception:
        return default


# -----------------------
# Build caches for users / dishes
# -----------------------
def build_user_cache(users_csv: str) -> Dict[int, Dict[str, Any]]:
    users = pd.read_csv(users_csv)
    cache: Dict[int, Dict[str, Any]] = {}

    # expected columns:
    # user_id, allergies, liked_tags, disliked_tags,
    # prefers_light_food, sweet_tooth, coffee_addict, microwave_trust
    for row in users.itertuples(index=False):
        uid = int(getattr(row, "user_id"))
        allergies = set(split_pipe(getattr(row, "allergies", "")))
        liked = set(split_pipe(getattr(row, "liked_tags", "")))
        disliked = set(split_pipe(getattr(row, "disliked_tags", "")))

        cache[uid] = {
            "allergies": allergies,
            "liked": liked,
            "disliked": disliked,
            "prefers_light_food": safe_int(getattr(row, "prefers_light_food", 0)),
            "sweet_tooth": safe_int(getattr(row, "sweet_tooth", 0)),
            "coffee_addict": safe_int(getattr(row, "coffee_addict", 0)),
            "microwave_trust": safe_int(getattr(row, "microwave_trust", 0)),
        }
    return cache


def build_dish_cache(dishes_csv: str) -> Dict[int, Dict[str, Any]]:
    dishes = pd.read_csv(dishes_csv)
    cache: Dict[int, Dict[str, Any]] = {}

    # expected columns:
    # dish_id, dish_name, category, calories, spicy, allergen_tags, tags
    for row in dishes.itertuples(index=False):
        did = int(getattr(row, "dish_id"))
        category = str(getattr(row, "category", "unknown"))
        calories = safe_float(getattr(row, "calories", 0.0))
        spicy = safe_int(getattr(row, "spicy", 0))
        allergen = set(split_pipe(getattr(row, "allergen_tags", "")))
        tags = set(split_pipe(getattr(row, "tags", "")))

        # простые derived-флаги
        is_dessert = 1 if category == "dessert" else 0
        is_drink = 1 if category == "drink" else 0
        is_breakfast = 1 if category == "breakfast" else 0

        cache[did] = {
            "category": category,
            "calories": calories,
            "spicy": spicy,
            "allergen": allergen,
            "tags": tags,
            "is_dessert": is_dessert,
            "is_drink": is_drink,
            "is_breakfast": is_breakfast,
        }
    return cache


# -----------------------
# Feature building (dict for hashing)
# -----------------------
def build_feature_dict(
    user: Dict[str, Any],
    dish: Dict[str, Any],
    ctx: Dict[str, Any],
) -> Dict[str, float]:
    """
    Возвращает dict[str, float] для FeatureHasher.
    """
    feats: Dict[str, float] = {}

    # --- context (categorical as one-hot via string keys)
    meal_slot = str(ctx.get("meal_slot", "unknown"))
    feats[f"meal_slot={meal_slot}"] = 1.0

    # day/hour-like: day is numeric but can also be bucketized
    day = safe_int(ctx.get("day", 0))
    feats[f"day={day}"] = 1.0  # one-hot по дню (1..18), нормально

    hang = safe_int(ctx.get("hangover_level", 0))
    feats[f"hangover={hang}"] = 1.0

    guests = safe_int(ctx.get("guests_count", 0))
    # добавим и numeric, и бины:
    feats["guests_count"] = float(guests)
    feats[f"guests_bin={min(guests, 6)}"] = 1.0

    diet_mode = safe_int(ctx.get("diet_mode", 0))
    feats["diet_mode"] = float(diet_mode)

    fridge_load = safe_float(ctx.get("fridge_load_pct", 0.0))
    feats["fridge_load_pct"] = fridge_load

    # --- user prefs (numeric)
    feats["u_prefers_light_food"] = float(user.get("prefers_light_food", 0))
    feats["u_sweet_tooth"] = float(user.get("sweet_tooth", 0))
    feats["u_coffee_addict"] = float(user.get("coffee_addict", 0))
    feats["u_microwave_trust"] = float(user.get("microwave_trust", 0))

    # --- dish properties
    cat = dish.get("category", "unknown")
    feats[f"d_category={cat}"] = 1.0
    feats["d_calories"] = float(dish.get("calories", 0.0))
    feats["d_spicy"] = float(dish.get("spicy", 0))
    feats["d_is_dessert"] = float(dish.get("is_dessert", 0))
    feats["d_is_drink"] = float(dish.get("is_drink", 0))
    feats["d_is_breakfast"] = float(dish.get("is_breakfast", 0))

    # --- interactions: tag overlaps & allergen conflicts
    u_liked = user.get("liked", set())
    u_disliked = user.get("disliked", set())
    u_allerg = user.get("allergies", set())

    d_tags = dish.get("tags", set())
    d_allerg = dish.get("allergen", set())

    liked_overlap = len(u_liked.intersection(d_tags))
    disliked_overlap = len(u_disliked.intersection(d_tags))
    allerg_conflict = len(u_allerg.intersection(d_allerg))

    feats["overlap_liked"] = float(liked_overlap)
    feats["overlap_disliked"] = float(disliked_overlap)
    feats["allergen_conflict"] = float(allerg_conflict)

    # штрафы/бусты в виде “ручных” взаимодействий
    # если аллергены есть — очень вероятно, что блюдо не выберут
    if allerg_conflict > 0:
        feats["HAS_ALLERGEN_CONFLICT"] = 1.0

    # сладкоежка * десерт
    feats["sweet_x_dessert"] = float(user.get("sweet_tooth", 0) * dish.get("is_dessert", 0))

    # диета: более вероятно выбрать не очень калорийное (грубый признак)
    # добавим инверсию калорий, чтобы диета могла её “поднять”
    cal = float(dish.get("calories", 0.0))
    feats["inv_calories"] = 1.0 / (1.0 + max(0.0, cal))

    feats["diet_x_invcal"] = float(diet_mode) * feats["inv_calories"]

    # контекстная совместимость: breakfast slot + breakfast category
    feats["slot_x_breakfast"] = 1.0 if (meal_slot == "breakfast" and dish.get("is_breakfast", 0) == 1) else 0.0

    # добавим сами теги блюда как разреженные признаки
    # (FeatureHasher спокойно это съест)
    for t in d_tags:
        feats[f"tag={t}"] = 1.0

    # и “любимые теги пользователя” как контекстные (можно и без, но иногда помогает)
    # чтобы не раздувать — ограничим до 30
    for t in list(u_liked)[:30]:
        feats[f"u_like={t}"] = 1.0

    return feats


# -----------------------
# Training data generator (streaming)
# -----------------------
def iter_train_pairs(
    train_csv: str,
    user_cache: Dict[int, Dict[str, Any]],
    dish_cache: Dict[int, Dict[str, Any]],
    neg_per_pos: int = 5,
    chunksize: int = 20000,
    seed: int = 42,
) -> Iterable[Tuple[List[Dict[str, float]], List[int]]]:
    """
    Стримит батчи (list[feature_dict], list[label]) без раздувания train в память.
    На каждый event: 1 positive + neg_per_pos negatives.
    """
    rng = random.Random(seed)
    cand_cols = [f"cand_{i:02d}" for i in range(1, 21)]

    reader = pd.read_csv(train_csv, chunksize=chunksize)
    for chunk in reader:
        X_dicts: List[Dict[str, float]] = []
        y: List[int] = []

        # заполним NaN в кандидатах
        for c in cand_cols:
            if c not in chunk.columns:
                raise ValueError(f"Missing column {c} in train.csv")
        if "target_dish_id" not in chunk.columns:
            raise ValueError("Missing target_dish_id in train.csv")

        for row in chunk.itertuples(index=False):
            uid = int(getattr(row, "user_id"))
            user = user_cache.get(uid)
            if user is None:
                # неизвестный юзер — скипаем (редко)
                continue

            ctx = {
                "day": getattr(row, "day"),
                "meal_slot": getattr(row, "meal_slot"),
                "hangover_level": getattr(row, "hangover_level"),
                "guests_count": getattr(row, "guests_count"),
                "diet_mode": getattr(row, "diet_mode"),
                "fridge_load_pct": getattr(row, "fridge_load_pct"),
            }

            target = int(getattr(row, "target_dish_id"))
            cands = []
            for c in cand_cols:
                v = getattr(row, c)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                cands.append(int(v))

            # гарантируем: target в кандидатах (по условию должен быть)
            if target not in cands:
                continue

            # positive
            dish_pos = dish_cache.get(target)
            if dish_pos is not None:
                X_dicts.append(build_feature_dict(user, dish_pos, ctx))
                y.append(1)

            # negatives: sample from candidates excluding target
            neg_pool = [d for d in cands if d != target]
            if not neg_pool:
                continue
            k = min(neg_per_pos, len(neg_pool))
            negs = rng.sample(neg_pool, k=k)

            for d in negs:
                dish_neg = dish_cache.get(d)
                if dish_neg is None:
                    continue
                X_dicts.append(build_feature_dict(user, dish_neg, ctx))
                y.append(0)

        if X_dicts:
            yield X_dicts, y


# -----------------------
# Train model
# -----------------------
def train_model(
    train_csv: str,
    users_csv: str,
    dishes_csv: str,
    neg_per_pos: int,
    n_features: int,
    chunksize: int,
    seed: int,
    passes: int = 1,
) -> Tuple[SGDClassifier, FeatureHasher]:
    user_cache = build_user_cache(users_csv)
    dish_cache = build_dish_cache(dishes_csv)

    hasher = FeatureHasher(n_features=n_features, input_type="dict", alternate_sign=False)

    # SGD логрег
    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-6,
        learning_rate="optimal",
        early_stopping=False,
        random_state=seed,
    )

    classes = np.array([0, 1], dtype=int)
    fitted = False

    for p in range(passes):
        t0 = time.time()
        n_batches = 0
        n_samples = 0

        for X_dicts, y in iter_train_pairs(
            train_csv=train_csv,
            user_cache=user_cache,
            dish_cache=dish_cache,
            neg_per_pos=neg_per_pos,
            chunksize=chunksize,
            seed=seed + 1000 * p,
        ):
            X = hasher.transform(X_dicts)
            y_arr = np.asarray(y, dtype=int)

            if not fitted:
                clf.partial_fit(X, y_arr, classes=classes)
                fitted = True
            else:
                clf.partial_fit(X, y_arr)

            n_batches += 1
            n_samples += len(y_arr)

            if n_batches % 50 == 0:
                dt = time.time() - t0
                print(f"pass {p+1}/{passes} | batches={n_batches} samples={n_samples} time={dt:.1f}s")

        dt = time.time() - t0
        print(f"Finished pass {p+1}/{passes}: batches={n_batches}, samples={n_samples}, time={dt:.1f}s")

    # вернем и кэши через замыкание? не нужно — кэши пересоберем на inference
    return clf, hasher


# -----------------------
# Inference: score 20 candidates and take top-5 unique
# -----------------------
def predict_top5(
    clf: SGDClassifier,
    hasher: FeatureHasher,
    test_csv: str,
    users_csv: str,
    dishes_csv: str,
    sample_sub_csv: str,
    out_csv: str,
    chunksize: int = 20000,
):
    user_cache = build_user_cache(users_csv)
    dish_cache = build_dish_cache(dishes_csv)

    cand_cols = [f"cand_{i:02d}" for i in range(1, 21)]

    sample = pd.read_csv(sample_sub_csv)
    if "query_id" not in sample.columns:
        raise ValueError("sample_submission must contain query_id")

    out_rows = []

    reader = pd.read_csv(test_csv, chunksize=chunksize)
    for chunk in reader:
        for c in cand_cols:
            if c not in chunk.columns:
                raise ValueError(f"Missing column {c} in test.csv")

        for row in chunk.itertuples(index=False):
            qid = int(getattr(row, "query_id"))
            uid = int(getattr(row, "user_id"))
            user = user_cache.get(uid)
            if user is None:
                # если юзера нет — просто выдадим первые 5 кандидатов
                cands = [int(getattr(row, c)) for c in cand_cols if not pd.isna(getattr(row, c))]
                cands = [d for d in cands if d in dish_cache]
                cands = list(dict.fromkeys(cands))  # unique preserving order
                recs = (cands + cands + cands)[:5] if cands else [1, 2, 3, 4, 5]
                out_rows.append([qid] + recs[:5])
                continue

            ctx = {
                "day": getattr(row, "day"),
                "meal_slot": getattr(row, "meal_slot"),
                "hangover_level": getattr(row, "hangover_level"),
                "guests_count": getattr(row, "guests_count"),
                "diet_mode": getattr(row, "diet_mode"),
                "fridge_load_pct": getattr(row, "fridge_load_pct"),
            }

            cands = []
            for c in cand_cols:
                v = getattr(row, c)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                d = int(v)
                if d in dish_cache:
                    cands.append(d)

            # уникальные кандидаты (на всякий)
            cands = list(dict.fromkeys(cands))
            if not cands:
                out_rows.append([qid, 1, 2, 3, 4, 5])
                continue

            # построим фичи на все кандидаты
            X_dicts = [build_feature_dict(user, dish_cache[d], ctx) for d in cands]
            X = hasher.transform(X_dicts)

            # predict_proba доступен для SGDClassifier с log_loss
            probs = clf.predict_proba(X)[:, 1]

            # сортировка по убыванию вероятности
            order = np.argsort(-probs)
            ranked = [cands[i] for i in order]

            # top-5 (уже уникальные)
            recs = ranked[:5]
            if len(recs) < 5:
                # добьем оставшимися кандидатами
                for d in cands:
                    if d not in recs:
                        recs.append(d)
                    if len(recs) == 5:
                        break
            # на крайний случай
            while len(recs) < 5:
                recs.append(recs[-1])

            out_rows.append([qid] + recs[:5])

    sub = pd.DataFrame(out_rows, columns=["query_id", "rec_1", "rec_2", "rec_3", "rec_4", "rec_5"])

    # порядок как в sample_submission
    sub = sample[["query_id"]].merge(sub, on="query_id", how="left")

    # если где-то не предсказали — заполним безопасно первыми 5 блюдами (валидность не гарантируется)
    for col in ["rec_1", "rec_2", "rec_3", "rec_4", "rec_5"]:
        sub[col] = sub[col].fillna(1).astype(int)

    sub.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    print(sub.head(10))


# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default="data/train_holodilnik.csv")
    parser.add_argument("--test_csv", type=str, default="data/test_holodilnik.csv")
    parser.add_argument("--users_csv", type=str, default="data/users.csv")
    parser.add_argument("--dishes_csv", type=str, default="data/dishes.csv")
    parser.add_argument("--sample_sub", type=str, default="data/sample_submission_holodilnik.csv")
    parser.add_argument("--out_csv", type=str, default="submission.csv")

    # --- Гиперпараметры ---
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--neg_per_pos", type=int, default=5)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--chunksize", type=int, default=20000)
    parser.add_argument("--n_features", type=int, default=2 ** 20)

    args = parser.parse_args()

    print("Training...")
    clf, hasher = train_model(
        train_csv=args.train_csv,
        users_csv=args.users_csv,
        dishes_csv=args.dishes_csv,
        neg_per_pos=args.neg_per_pos,
        n_features=args.n_features,
        chunksize=args.chunksize,
        seed=args.seed,
        passes=args.passes,
    )

    print("Predicting...")
    predict_top5(
        clf=clf,
        hasher=hasher,
        test_csv=args.test_csv,
        users_csv=args.users_csv,
        dishes_csv=args.dishes_csv,
        sample_sub_csv=args.sample_sub,
        out_csv=args.out_csv,
        chunksize=args.chunksize,
    )


if __name__ == "__main__":
    main()
