from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")

TRAIN_PATH = DATA_DIR / "train_holodilnik.csv"
TEST_PATH = DATA_DIR / "test_holodilnik.csv"
USERS_PATH = DATA_DIR / "users.csv"
DISHES_PATH = DATA_DIR / "dishes.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission_holodilnik.csv"

CAND_COLS = [f"cand_{i:02d}" for i in range(1, 21)]


def split_pipe(s: str):
    """mayo|peas -> set(['mayo','peas']); пусто/NaN -> пустое множество"""
    if pd.isna(s) or s == "":
        return set()
    return set(s.split("|"))


def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    users = pd.read_csv(USERS_PATH)
    dishes = pd.read_csv(DISHES_PATH)

    # предрасчёт множеств (для быстрых пересечений)
    users["allergies_set"] = users["allergies"].apply(split_pipe)
    users["liked_set"] = users["liked_tags"].apply(split_pipe)
    users["disliked_set"] = users["disliked_tags"].apply(split_pipe)

    dishes["allergen_set"] = dishes["allergen_tags"].apply(split_pipe)
    dishes["tags_set"] = dishes["tags"].apply(split_pipe)

    return train, test, users, dishes


def to_long_format(df: pd.DataFrame, *, is_train: bool):
    """
    Из wide (cand_01..cand_20) делаем long:
    columns: [row_id, user_id, context..., cand_pos, dish_id, label?]
    """
    base_cols = [c for c in df.columns if c not in CAND_COLS]
    if is_train:
        # target_dish_id есть в base_cols
        pass

    # melt: одна строка -> 20 строк
    long_df = df.melt(
        id_vars=base_cols,
        value_vars=CAND_COLS,
        var_name="cand_pos",
        value_name="dish_id"
    )

    # позиция кандидата: cand_01 -> 1, cand_20 -> 20
    long_df["cand_pos"] = long_df["cand_pos"].str[-2:].astype(int)

    if is_train:
        long_df["label"] = (long_df["dish_id"] == long_df["target_dish_id"]).astype(int)

    return long_df


def add_features(long_df: pd.DataFrame, users: pd.DataFrame, dishes: pd.DataFrame):
    # присоединим user- и dish-таблицы
    df = long_df.merge(users, on="user_id", how="left")
    df = df.merge(dishes, on="dish_id", how="left")

    # --- быстрые численные признаки по множествам ---
    # пересечение liked_tags и dish.tags
    def inter_size(a, b):
        if not isinstance(a, set) or not isinstance(b, set):
            return 0
        return len(a & b)

    def union_size(a, b):
        if not isinstance(a, set) or not isinstance(b, set):
            return 0
        return len(a | b)

    df["liked_overlap"] = [
        inter_size(u, t) for u, t in zip(df["liked_set"], df["tags_set"])
    ]
    df["disliked_overlap"] = [
        inter_size(u, t) for u, t in zip(df["disliked_set"], df["tags_set"])
    ]

    # Jaccard liked vs dish tags (нормировка)
    df["liked_jaccard"] = [
        (inter_size(u, t) / (union_size(u, t) + 1e-9)) for u, t in zip(df["liked_set"], df["tags_set"])
    ]

    # аллергены: конфликт?
    df["allergen_conflict"] = [
        1 if inter_size(a, al) > 0 else 0 for a, al in zip(df["allergies_set"], df["allergen_set"])
    ]

    # штраф за аллерген: можно жёстко “убивать” скор на инференсе,
    # но на обучении лучше дать модели это понять через фичу.

    # --- контекстные взаимодействия ---
    # диета: блюдо "полегче" ~ низкие калории
    df["diet_x_calories"] = df["diet_mode"] * df["calories"]

    # сладкоежка × категория/теги
    df["sweet_x_is_dessert"] = df["sweet_tooth"] * (df["category"] == "dessert").astype(int)
    df["sweet_x_tag_sweet"] = df["sweet_tooth"] * df["tags_set"].apply(lambda s: 1 if ("sweet" in s) else 0)

    # кофеин-зависимость: напитки/кофе теги
    df["coffee_x_drink"] = df["coffee_addict"] * (df["category"] == "drink").astype(int)
    df["coffee_x_tag_coffee"] = df["coffee_addict"] * df["tags_set"].apply(
        lambda s: 1 if ("coffee" in s or "caffeine" in s) else 0)

    # похмелье: условно может тянуть на суп/жирное/солёное/острое (модель разберётся)
    df["hangover_x_soup"] = df["hangover_level"] * (df["category"] == "soup").astype(int)
    df["hangover_x_spicy"] = df["hangover_level"] * df["spicy"]

    # гости: “общие” блюда/снеки (тоже скорее через теги)
    df["guests_x_snack"] = df["guests_count"] * (df["category"] == "snack").astype(int)

    # микроволновка: иногда влияет на выбор (если есть теги типа microwave/quick) — если таких тегов нет, всё равно пусть будет
    df["micro_x_quick_tag"] = df["microwave_trust"] * df["tags_set"].apply(
        lambda s: 1 if ("quick" in s or "microwave" in s) else 0)

    return df


def build_feature_columns(df: pd.DataFrame):
    # базовые контекстные и справочные поля
    cat_features = [
        "meal_slot",
        "category",
    ]
    # можно добавить dish_name, но обычно это шум; лучше не надо

    num_features = [
        "day",
        "hangover_level",
        "guests_count",
        "diet_mode",
        "fridge_load_pct",
        "cand_pos",

        "calories",
        "spicy",

        "prefers_light_food",
        "sweet_tooth",
        "coffee_addict",
        "microwave_trust",

        "liked_overlap",
        "disliked_overlap",
        "liked_jaccard",
        "allergen_conflict",

        "diet_x_calories",
        "sweet_x_is_dessert",
        "sweet_x_tag_sweet",
        "coffee_x_drink",
        "coffee_x_tag_coffee",
        "hangover_x_soup",
        "hangover_x_spicy",
        "guests_x_snack",
        "micro_x_quick_tag",
    ]
    features = cat_features + num_features
    return features, cat_features


from catboost import CatBoostRanker, Pool


def train_ranker(train_wide, users, dishes):
    train_long = to_long_format(train_wide, is_train=True)
    train_feat = add_features(train_long, users, dishes)

    # split (пример: по day)
    train_part = train_feat[train_feat["day"] <= 14].copy()
    valid_part = train_feat[train_feat["day"] >= 15].copy()

    # !!! КРИТИЧНО: группировки должны идти подряд
    train_part = train_part.sort_values(["event_id", "cand_pos"]).reset_index(drop=True)
    valid_part = valid_part.sort_values(["event_id", "cand_pos"]).reset_index(drop=True)

    features, cat_features = build_feature_columns(train_feat)

    train_pool = Pool(
        data=train_part[features],
        label=train_part["label"],
        group_id=train_part["event_id"],  # берем после сортировки
        cat_features=cat_features
    )
    valid_pool = Pool(
        data=valid_part[features],
        label=valid_part["label"],
        group_id=valid_part["event_id"],  # берем после сортировки
        cat_features=cat_features
    )

    model = CatBoostRanker(
        loss_function="YetiRank",
        eval_metric="NDCG:top=5",
        iterations=3000,
        learning_rate=0.05,
        depth=8,
        random_seed=42,
        verbose=200,
        od_type="Iter",
        od_wait=200,
        l2_leaf_reg=5.0,

        task_type="GPU",
        devices="0"
    )

    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    return model


def predict_top5(model, test_wide, users, dishes):
    test_long = to_long_format(test_wide, is_train=False)
    test_feat = add_features(test_long, users, dishes)

    features, cat_features = build_feature_columns(test_feat)

    # !!! КРИТИЧНО: query_id должны идти блоками
    test_feat = test_feat.sort_values(["query_id", "cand_pos"]).reset_index(drop=True)

    test_pool = Pool(
        data=test_feat[features],
        group_id=test_feat["query_id"],   # берём после сортировки
        cat_features=cat_features
    )

    test_feat["score"] = model.predict(test_pool)

    # опционально: жёстко штрафуем аллерген
    test_feat.loc[test_feat["allergen_conflict"] == 1, "score"] -= 1e6

    # собираем top-5
    recs = []
    for qid, part in test_feat.groupby("query_id", sort=False):
        part_sorted = part.sort_values("score", ascending=False)

        top_dishes = []
        for dish_id in part_sorted["dish_id"].tolist():
            dish_id = int(dish_id)
            if dish_id not in top_dishes:
                top_dishes.append(dish_id)
            if len(top_dishes) == 5:
                break

        # добивка (на всякий случай)
        if len(top_dishes) < 5:
            for dish_id in part_sorted["dish_id"].tolist():
                dish_id = int(dish_id)
                if dish_id not in top_dishes:
                    top_dishes.append(dish_id)
                if len(top_dishes) == 5:
                    break

        recs.append([qid] + top_dishes)

    sub = pd.DataFrame(recs, columns=["query_id", "rec_1", "rec_2", "rec_3", "rec_4", "rec_5"])
    return sub


def main():
    train, test, users, dishes = load_data()

    model = train_ranker(train, users, dishes)

    submission = predict_top5(model, test, users, dishes)

    out_path = Path("submission.csv")
    submission.to_csv(out_path, index=False)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
