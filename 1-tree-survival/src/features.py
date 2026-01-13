from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


TARGET_COL = "survived_to_18jan"
ID_COL = "apartment_id"


def get_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """
    Returns:
      - features (excluding target & apartment_id)
      - categorical feature names
      - numeric feature names
    """
    cols = list(df.columns)
    features = [c for c in cols if c not in [TARGET_COL, ID_COL]]

    cat_cols = [c for c in features if df[c].dtype == "object"]
    num_cols = [c for c in features if c not in cat_cols]
    return features, cat_cols, num_cols


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lightweight, safe feature engineering.
    CatBoost can handle NaNs in numeric; for categorical we will fill later.
    """
    out = df.copy()

    # Days from cut to Jan 18 (Jan 1 -> Jan 18 is 17 days)
    if "cut_days_before_jan1" in out.columns:
        out["cut_days_to_18jan"] = out["cut_days_before_jan1"] + 17

    # Garland heat proxy: non-LED garlands generally warm more (rough heuristic)
    if "garland_hours_per_day" in out.columns and "led_garland" in out.columns:
        # If led_garland==1 -> factor 0.6, else 1.0
        out["garland_heat_proxy"] = out["garland_hours_per_day"] * np.where(
            out["led_garland"] == 1, 0.6, 1.0
        )

    # Radiator proximity inverse (closer => higher effect)
    if "radiator_distance_m" in out.columns:
        out["radiator_inverse_dist"] = 1.0 / (out["radiator_distance_m"] + 0.3)

    # Temp-humidity interaction
    if "room_temp_c" in out.columns and "humidity_pct" in out.columns:
        out["temp_x_humidity"] = out["room_temp_c"] * out["humidity_pct"]

    # Watering + mist together
    if "waterings_per_week" in out.columns and "mist_spray" in out.columns:
        out["watering_plus_mist"] = out["waterings_per_week"] + out["mist_spray"].fillna(0)

    # Ornaments load per height
    if "ornaments_weight_kg" in out.columns and "tree_height_cm" in out.columns:
        out["ornaments_per_100cm"] = out["ornaments_weight_kg"] / (out["tree_height_cm"] / 100.0 + 0.5)

    # Per-area ventilation intensity
    if "window_ventilation_per_day" in out.columns and "apartment_area_m2" in out.columns:
        out["ventilation_per_10m2"] = out["window_ventilation_per_day"] / (out["apartment_area_m2"] / 10.0 + 0.5)

    return out
