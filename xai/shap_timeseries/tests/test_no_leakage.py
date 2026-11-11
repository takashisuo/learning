"""特徴量が未来情報を含まないことの確認。

時点 t の特徴量を、t 以降のデータを書き換えたデータで再計算しても変わらないことを確かめる。
- 目的変数: t 自身を含めて t 以降を書き換える(t の特徴量に y_t が入っていてもリーク)。
- 外生変数: t より後を書き換える(t の外生変数は予報済み = 既知と仮定しているため t は対象外)。
"""
import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.data import load_daily, time_split
from src.features import build_features

CFG = load_config()


@pytest.fixture(scope="module")
def df():
    return load_daily(CFG)


@pytest.fixture(scope="module")
def base(df):
    return build_features(df, CFG).X


def _check_points(X: pd.DataFrame) -> list[pd.Timestamp]:
    rng = np.random.default_rng(CFG["seed"])
    idx = X.index[rng.choice(len(X) - 1, size=20, replace=False)]
    return sorted(idx) + [X.index[0], X.index[-2]]


def test_future_target_does_not_change_features(df, base):
    target = CFG["data"]["target"]
    rng = np.random.default_rng(0)
    for t in _check_points(base):
        mod = df.copy()
        future = mod.index >= t
        mod.loc[future, target] = rng.integers(0, 10_000, future.sum())
        X = build_features(mod, CFG).X
        pd.testing.assert_series_equal(X.loc[t], base.loc[t], check_names=False)


def test_future_exog_does_not_change_features(df, base):
    exog = CFG["features"]["exog"] + ["holiday"]
    rng = np.random.default_rng(1)
    for t in _check_points(base):
        mod = df.copy()
        future = mod.index > t
        for c in exog:
            mod.loc[future, c] = rng.permutation(mod.loc[future, c].to_numpy())
        X = build_features(mod, CFG).X
        pd.testing.assert_series_equal(X.loc[t], base.loc[t], check_names=False)


def test_ar_features_use_only_past_values(df, base):
    """lag_k は k日前の実績、移動統計は前日までの窓と一致する。"""
    y = df[CFG["data"]["target"]]
    t = base.index[100]
    pos = y.index.get_loc(t)
    for k in CFG["features"]["ar"]["lags"]:
        assert base.loc[t, f"lag_{k}"] == y.iloc[pos - k]
    for w in CFG["features"]["ar"]["rolling"]["windows"]:
        assert np.isclose(base.loc[t, f"roll{w}_mean"], y.iloc[pos - w:pos].mean())
        assert np.isclose(base.loc[t, f"roll{w}_std"], y.iloc[pos - w:pos].std())


def test_split_is_time_ordered(df):
    s = time_split(df.index, CFG["split"]["test_ratio"])
    assert s.train_index.max() < s.test_index.min()
    assert len(s.train_index) + len(s.test_index) == len(df)
