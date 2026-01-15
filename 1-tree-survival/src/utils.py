from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def pretty_params(d: Dict[str, Any]) -> str:
    lines = []
    for k in sorted(d.keys()):
        lines.append(f"{k}: {d[k]}")
    return "\n".join(lines)


@dataclass(frozen=True)
class Paths:
    train_path: str
    test_path: str
    sample_submission_path: str
    outputs_dir: str
