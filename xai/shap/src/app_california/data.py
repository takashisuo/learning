"""データ読込(California Housing / 動作確認用の diabetes)。"""
from __future__ import annotations

from sklearn.datasets import fetch_california_housing, load_diabetes

from src.xai_platform.data import Dataset, set_seed, split_dataset

from .config import SEED, TARGET_NAME, TEST_SIZE


def load_california() -> Dataset:
    set_seed(SEED)
    bunch = fetch_california_housing(as_frame=True)
    return split_dataset(bunch.data, bunch.target.rename(TARGET_NAME), "california_housing",
                         TEST_SIZE, SEED)


def load_diabetes_dataset() -> Dataset:
    bunch = load_diabetes(as_frame=True)
    return split_dataset(bunch.data, bunch.target.rename("disease_progression"), "diabetes",
                         TEST_SIZE, SEED)
