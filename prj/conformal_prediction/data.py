"""サンプルの時系列データを作成するモジュール。"""

import numpy as np
import pandas as pd


def generate_sample_timeseries(
    n_days: int = 730,
    start_date: str = "2023-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """トレンド・週次/年次季節性・ノイズを持つ日次時系列データを生成する。

    Returns
    -------
    pd.DataFrame: columns = ["date", "y"]
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start_date, periods=n_days, freq="D")
    t = np.arange(n_days)

    trend = 0.05 * t
    weekly_seasonality = 10 * np.sin(2 * np.pi * t / 7)
    noise = rng.normal(loc=0.0, scale=5.0, size=n_days)

    y = 100 + trend + weekly_seasonality + noise

    return pd.DataFrame({"date": dates, "y": y})


def create_features(
    df: pd.DataFrame,
    lags: tuple[int, ...] = (1, 2, 3, 7, 14),
    rolling_windows: tuple[int, ...] = (7, 14),
) -> pd.DataFrame:
    """機械学習モデル用にラグ特徴量・移動平均・曜日特徴量を作成する。

    予測時点で未来情報を使わないよう、全ての特徴量は shift(1) 以降の値のみを使う。
    """
    df = df.sort_values("date").reset_index(drop=True).copy()

    for lag in lags:
        df[f"lag_{lag}"] = df["y"].shift(lag)

    for window in rolling_windows:
        df[f"rolling_mean_{window}"] = df["y"].shift(1).rolling(window).mean()
        df[f"rolling_std_{window}"] = df["y"].shift(1).rolling(window).std()

    df["dayofweek"] = df["date"].dt.dayofweek
    df["day_index"] = np.arange(len(df))

    df = df.dropna().reset_index(drop=True)
    return df
