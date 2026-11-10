"""
目的: skrub サンプルで共通利用する titanic データの読み込み処理。

- OpenML から titanic データを取得し、ローカル (data/titanic.csv) にキャッシュする
- skrub.Cleaner の効果を見せるため、意図的に「汚した」データも生成できる

動作確認: Python 3.12 / skrub 0.10.1 / scikit-learn 1.9.1 / pandas 3.0.6
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
CACHE_PATH = DATA_DIR / "titanic.csv"

TARGET = "survived"
# 目的変数と強く結びつく「答えを含む」列 (救命ボート番号 / 遺体識別番号)
LEAKAGE_COLUMNS = ["boat", "body"]


def load_titanic(cache_path: Path = CACHE_PATH) -> pd.DataFrame:
    """titanic データを読み込む。初回のみ OpenML からダウンロードして CSV に保存する。"""
    if cache_path.exists():
        return pd.read_csv(cache_path)

    from sklearn.datasets import fetch_openml

    df = fetch_openml("titanic", version=1, as_frame=True).frame
    df[TARGET] = df[TARGET].astype(int)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return pd.read_csv(cache_path)


def make_dirty_titanic(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """
    実務でありがちな「汚いデータ」を再現する。

    - 数値列が文字列として保存されている (fare)
    - 欠損値が "N/A" / "" などの文字列で表現されている (age)
    - 日付が文字列で入っている (boarding_date)
    - 全行同じ値の意味のない列 (ship_name)
    - ほぼ全て欠損の列 (memo)
    """
    rng = np.random.default_rng(seed)
    dirty = df.copy()

    dirty["fare"] = dirty["fare"].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    dirty["age"] = dirty["age"].map(lambda v: "N/A" if pd.isna(v) else str(v))

    dates = pd.Timestamp("1912-04-01") + pd.to_timedelta(
        rng.integers(0, 10, len(dirty)), unit="D"
    )
    dirty["boarding_date"] = dates.strftime("%Y-%m-%d")

    dirty["ship_name"] = "Titanic"
    dirty["memo"] = np.where(rng.random(len(dirty)) < 0.01, "要確認", None)
    return dirty.astype({"fare": "str", "age": "str"})
