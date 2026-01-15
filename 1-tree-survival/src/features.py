from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


TARGET_COL = "survived_to_18jan"
ID_COL = "apartment_id"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Safe, light FE. Doesn't need fitting, works on train/test одинаково.
    """
    out = df.copy()

    # cut days to Jan 18
    if "cut_days_before_jan1" in out.columns:
        out["cut_days_to_18jan"] = out["cut_days_before_jan1"] + 17

    # heat proxy from garland (non-led tends to heat more)
    if "garland_hours_per_day" in out.columns and "led_garland" in out.columns:
        out["garland_heat_proxy"] = out["garland_hours_per_day"] * np.where(
            out["led_garland"] == 1, 0.6, 1.0
        )

    # radiator inverse distance
    if "radiator_distance_m" in out.columns:
        out["radiator_inverse_dist"] = 1.0 / (out["radiator_distance_m"] + 0.3)

    # temp-humidity interaction
    if "room_temp_c" in out.columns and "humidity_pct" in out.columns:
        out["temp_x_humidity"] = out["room_temp_c"] * out["humidity_pct"]
        out["dryness_index"] = out["room_temp_c"] / (out["humidity_pct"] + 1.0)

    # watering + mist
    if "waterings_per_week" in out.columns and "mist_spray" in out.columns:
        out["watering_plus_mist"] = out["waterings_per_week"].fillna(0) + out["mist_spray"].fillna(0)
        out["care_score"] = out["waterings_per_week"].fillna(0) * 1.0 + out["mist_spray"].fillna(0) * 2.0

    # ornaments load per height
    if "ornaments_weight_kg" in out.columns and "tree_height_cm" in out.columns:
        out["ornaments_per_100cm"] = out["ornaments_weight_kg"] / (out["tree_height_cm"] / 100.0 + 0.5)

    # ventilation intensity per area
    if "window_ventilation_per_day" in out.columns and "apartment_area_m2" in out.columns:
        out["ventilation_per_10m2"] = out["window_ventilation_per_day"] / (out["apartment_area_m2"] / 10.0 + 0.5)

    # draft proxy
    if "window_ventilation_per_day" in out.columns and "corner_apartment" in out.columns:
        out["draft_proxy"] = out["window_ventilation_per_day"] * (1.0 + out["corner_apartment"].fillna(0))

    return out


def get_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    cols = list(df.columns)
    features = [c for c in cols if c not in [TARGET_COL, ID_COL]]

    cat_cols = [c for c in features if df[c].dtype == "object"]
    num_cols = [c for c in features if c not in cat_cols]
    return features, cat_cols, num_cols
