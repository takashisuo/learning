"""データ読込(元の単位への変換を含む)と時間順分割。"""
from __future__ import annotations

import zipfile
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from .config import ROOT


class DataNotFoundError(FileNotFoundError):
    pass


def _read_raw(cfg: dict) -> pd.DataFrame:
    d = cfg["data"]
    raw = ROOT / d["raw_dir"]
    if (raw / d["csv"]).exists():
        return pd.read_csv(raw / d["csv"])
    if (raw / d["zip"]).exists():
        with zipfile.ZipFile(raw / d["zip"]) as z, z.open(d["csv"]) as f:
            return pd.read_csv(f)
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        raise DataNotFoundError(
            f"{raw / d['csv']} も {raw / d['zip']} も無く、ucimlrepo も未インストールです。"
            "UCI Bike Sharing Dataset の day.csv を data/raw/ に手動で配置してください。") from None
    # ucimlrepo の day.csv 相当(features + targets)。日次版かどうかは呼び出し側で確認する
    repo = fetch_ucirepo(id=d["uci_id"])
    return pd.concat([repo.data.features, repo.data.targets], axis=1)


def load_daily(cfg: dict) -> pd.DataFrame:
    """日次データを日付 index で返す。リーク列を除き、正規化済みの気象値を元の単位に戻す。"""
    d = cfg["data"]
    df = _read_raw(cfg)
    df[d["date_col"]] = pd.to_datetime(df[d["date_col"]])
    df = df.set_index(d["date_col"]).sort_index()
    df.index.name = "date"
    df = df.drop(columns=[c for c in d["drop_cols"] if c in df.columns])
    for src, spec in d["rescale"].items():
        df[spec["name"]] = df.pop(src) * spec["scale"]
    assert df.index.to_series().diff().dropna().eq(pd.Timedelta(days=1)).all(), "日付に欠落がある"
    return df


@dataclass
class TimeSplit:
    """時間順の train/test 分割。test は末尾 test_ratio の期間。"""
    train_index: pd.DatetimeIndex
    test_index: pd.DatetimeIndex

    @property
    def test_start(self) -> pd.Timestamp:
        return self.test_index[0]


def time_split(index: pd.DatetimeIndex, test_ratio: float) -> TimeSplit:
    """シャッフルせず、末尾 test_ratio を test にする。"""
    n_test = int(round(len(index) * test_ratio))
    return TimeSplit(index[:-n_test], index[-n_test:])


def cv_splitter(n_splits: int) -> TimeSeriesSplit:
    """チューニング用の拡張窓 CV。"""
    return TimeSeriesSplit(n_splits=n_splits)
