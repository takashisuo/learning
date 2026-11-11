"""データセットの入れ物と train/test 分割。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .provenance import produces_table


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


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def split_dataset(X: pd.DataFrame, y: pd.Series, name: str, test_size: float,
                  seed: int) -> Dataset:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed)
    return Dataset(X_tr, X_te, y_tr, y_te, name)


@produces_table
def description_table(descriptions: dict[str, str]) -> pd.DataFrame:
    """特徴量名 → 説明 の辞書を表にする。"""
    return pd.DataFrame({"特徴量": list(descriptions), "説明": list(descriptions.values())})
