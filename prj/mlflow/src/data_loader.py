"""Titanicデータセットの取得・前処理を行うモジュール。

- 第一候補: seaborn.load_dataset("titanic") によるオンライン取得
- フォールバック: data/titanic.csv のローカルファイル読み込み
- パス操作はすべて pathlib.Path を使用し、OS依存の絶対パス/ドライブレターは扱わない。
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# このファイル(data_loader.py)からの相対位置で data/ を解決する。
# 絶対パスの決め打ちはしない。
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LOCAL_CSV_PATH = DATA_DIR / "titanic.csv"

FEATURE_COLUMNS = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
TARGET_COLUMN = "survived"


def _load_raw_titanic() -> pd.DataFrame:
    """Titanicデータセットを取得する。

    seaborn経由の取得を優先し、失敗した場合はローカルCSV(data/titanic.csv)に
    フォールバックする。ローカルCSVも存在しない場合は、seabornから取得した
    データを data/titanic.csv として保存し、以降オフラインでも使えるようにする。
    """
    try:
        logger.info("seaborn.load_dataset('titanic') によるデータ取得を試みます。")
        df = sns.load_dataset("titanic")
        if df is None or df.empty:
            raise ValueError("seabornから取得したデータが空です。")

        # 次回以降オフラインでも動作するようにローカルへキャッシュしておく。
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not LOCAL_CSV_PATH.exists():
            df.to_csv(LOCAL_CSV_PATH, index=False)
            logger.info("取得したデータを %s にキャッシュしました。", LOCAL_CSV_PATH)

        return df
    except Exception as exc:  # noqa: BLE001 - オフライン環境等、広くフォールバックしたい
        logger.warning(
            "seabornからのデータ取得に失敗しました(%s)。"
            "ローカルファイル %s へのフォールバックを試みます。",
            exc,
            LOCAL_CSV_PATH,
        )

    if not LOCAL_CSV_PATH.exists():
        raise FileNotFoundError(
            f"seabornからの取得に失敗し、フォールバック先の {LOCAL_CSV_PATH} も"
            "見つかりませんでした。data/titanic.csv を配置するか、"
            "ネットワーク接続を確認してください。"
        )

    logger.info("ローカルファイル %s からデータを読み込みます。", LOCAL_CSV_PATH)
    return pd.read_csv(LOCAL_CSV_PATH)


def load_titanic_data() -> pd.DataFrame:
    """前処理済みのTitanicデータセットを返す。

    - 学習に使う特徴量 + 目的変数のみに絞り込む
    - 欠損値の簡単な補完
    - カテゴリ変数(sex, embarked)の数値エンコーディング
    """
    df = _load_raw_titanic()

    columns = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c in df.columns]
    df = df[columns].copy()

    # 欠損値補完(サンプルコードのため簡易的な方法にとどめる)
    if "age" in df.columns:
        df["age"] = df["age"].fillna(df["age"].median())
    if "fare" in df.columns:
        df["fare"] = df["fare"].fillna(df["fare"].median())
    if "embarked" in df.columns:
        df["embarked"] = df["embarked"].fillna(df["embarked"].mode().iloc[0])

    # カテゴリ変数の数値エンコーディング
    if "sex" in df.columns:
        df["sex"] = df["sex"].map({"male": 0, "female": 1})
    if "embarked" in df.columns:
        df["embarked"] = df["embarked"].astype("category").cat.codes

    df = df.dropna()

    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """特徴量と目的変数に分割する。"""
    x = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return x, y
