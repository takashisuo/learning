"""特徴量の生成(自己回帰・カレンダー・トレンド・外生変数)とグループ定義。

リーク防止のルール:
- 自己回帰特徴量は目的変数を shift(1) してから作る。時点 t の行には t-1 以前の目的変数しか入らない。
- 外生変数は時点 t の値をそのまま使う(予測時点で既知 = 予報済みと仮定。report に明記する)。
- カレンダー・トレンドは日付から決まるので未来も既知。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FeatureSet:
    X: pd.DataFrame                  # 特徴量(ラグで欠ける先頭行は除外済み)
    y: pd.Series                     # 目的変数(元スケール)
    groups: dict[str, list[str]]     # グループ名 → 特徴量名
    exog: list[str]                  # What-if で動かしてよい外生変数

    @property
    def group_of(self) -> dict[str, str]:
        return {f: g for g, fs in self.groups.items() for f in fs}


def ar_features(y: pd.Series, lags: list[int], windows: list[int],
                stats: list[str]) -> pd.DataFrame:
    """ラグと移動統計。すべて前日までの値から計算する。"""
    past = y.shift(1)  # 時点 t から見える最新は t-1
    out = {f"lag_{k}": y.shift(k) for k in lags}
    for w in windows:
        for s in stats:
            out[f"roll{w}_{s}"] = getattr(past.rolling(w), s)()
    return pd.DataFrame(out, index=y.index)


def calendar_features(index: pd.DatetimeIndex, holiday: pd.Series,
                      names: list[str]) -> pd.DataFrame:
    doy = index.dayofyear.to_numpy()
    days_in_year = np.where(index.is_leap_year, 366, 365)
    weekday = (index.dayofweek.to_numpy() + 1) % 7  # 元データと同じ 0=日 … 6=土
    every = {
        "weekday": weekday,
        "month": index.month.to_numpy(),
        "doy_sin": np.sin(2 * np.pi * doy / days_in_year),
        "doy_cos": np.cos(2 * np.pi * doy / days_in_year),
        "weekend": np.isin(weekday, [0, 6]).astype(int),
        "holiday": holiday.to_numpy(),
    }
    every["workingday"] = ((every["weekend"] == 0) & (every["holiday"] == 0)).astype(int)
    return pd.DataFrame({n: every[n] for n in names}, index=index)


def trend_features(index: pd.DatetimeIndex, names: list[str]) -> pd.DataFrame:
    every = {"t": (index - index[0]).days.to_numpy()}
    return pd.DataFrame({n: every[n] for n in names}, index=index)


def build_features(df: pd.DataFrame, cfg: dict) -> FeatureSet:
    """config の定義どおりに特徴量を作り、ラグで欠けた先頭行を落とす。"""
    fc, target = cfg["features"], cfg["data"]["target"]
    y = df[target].astype(float)
    parts = {
        "ar": ar_features(y, fc["ar"]["lags"], fc["ar"]["rolling"]["windows"],
                          fc["ar"]["rolling"]["stats"]),
        "calendar": calendar_features(df.index, df["holiday"], fc["calendar"]),
        "trend": trend_features(df.index, fc["trend"]),
        "exog": df[fc["exog"]].astype(float),
    }
    X = pd.concat(parts.values(), axis=1).astype(float).dropna()
    groups = {cfg["groups"][k]: list(p.columns) for k, p in parts.items()}
    return FeatureSet(X, y.loc[X.index], groups, list(parts["exog"].columns))
