"""データ読込・分割。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.model_selection import train_test_split

SEED = 42
TEST_SIZE = 0.2

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"

TARGET_NAME = "MedHouseVal"
TARGET_UNIT = "10万ドル"

# 特徴量の日本語説明(初出時の説明表に使う)
FEATURE_DESCRIPTIONS = {
    "MedInc": "地区の世帯収入中央値(単位: 1万ドル)",
    "HouseAge": "地区の住宅築年数の中央値(年)",
    "AveRooms": "1世帯あたり平均部屋数",
    "AveBedrms": "1世帯あたり平均寝室数",
    "Population": "地区の人口",
    "AveOccup": "1世帯あたり平均居住人数",
    "Latitude": "緯度(大きいほど北)",
    "Longitude": "経度(大きいほど東=内陸)",
}


@dataclass
class Dataset:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    name: str

    @property
    def feature_names(self) -> list[str]:
        return list(self.X_train.columns)


def ensure_dirs() -> None:
    for d in (FIGURES, TABLES):
        d.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)


def load_california() -> Dataset:
    """California Housing を読み込み、80/20 に分割する。"""
    set_seed()
    bunch = fetch_california_housing(as_frame=True)
    X = bunch.data
    y = bunch.target.rename(TARGET_NAME)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
    return Dataset(X_tr, X_te, y_tr, y_te, "california_housing")


def load_diabetes_dataset() -> Dataset:
    """動作確認用の diabetes データ。"""
    bunch = load_diabetes(as_frame=True)
    X = bunch.data
    y = bunch.target.rename("disease_progression")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
    return Dataset(X_tr, X_te, y_tr, y_te, "diabetes")


def feature_description_table() -> pd.DataFrame:
    return pd.DataFrame(
        {"特徴量": list(FEATURE_DESCRIPTIONS), "説明": list(FEATURE_DESCRIPTIONS.values())}
    )
