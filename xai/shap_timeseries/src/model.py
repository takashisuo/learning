"""目的変数の変換・学習(TimeSeriesSplit でのチューニング)・評価・Naive 比較。"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import cv_splitter
from .xai_platform import produces_table

LEVEL_FEATURE = "roll7_mean"  # level 変換で差し引く「直近水準」


# ---------------------------------------------------------------- 目的変数の変換
def forward(kind: str, y: pd.Series, X: pd.DataFrame) -> pd.Series:
    """元スケール → モデルが学習するスケール。"""
    if kind == "none":
        return y
    if kind == "log":
        return np.log1p(y)
    if kind == "level":
        return y - X[LEVEL_FEATURE]
    raise ValueError(kind)


def inverse(kind: str, f: np.ndarray, X: pd.DataFrame) -> np.ndarray:
    """モデル出力 → 元スケールの予測値。"""
    f = np.asarray(f, dtype=float)
    if kind == "none":
        return f
    if kind == "log":
        return np.expm1(f)
    if kind == "level":
        return f + X[LEVEL_FEATURE].to_numpy()
    raise ValueError(kind)


@dataclass
class TSModel:
    """推定器 + 目的変数の変換。SHAP は estimator(モデルスケール)に対して計算する。"""
    name: str
    estimator: object
    transform: str
    params: dict

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TSModel":
        self.estimator.fit(X, forward(self.transform, y, X))
        return self

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return inverse(self.transform, self.predict_raw(X), X)


def make_estimator(kind: str, params: dict, cfg: dict):
    seed = cfg["seed"]
    if kind == "lgbm":
        return LGBMRegressor(**cfg["model"]["lgbm"]["fixed"], **params, random_state=seed)
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(**params))
    raise ValueError(kind)


# ---------------------------------------------------------------- 評価指標
def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    err = pred - y
    return {"RMSE": float(np.sqrt(np.mean(err ** 2))),
            "MAE": float(np.mean(np.abs(err))),
            "MAPE(%)": float(np.mean(np.abs(err) / np.abs(y)) * 100),
            "sMAPE(%)": float(np.mean(2 * np.abs(err) / (np.abs(y) + np.abs(pred))) * 100)}


def _rmse(y, pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(pred)) ** 2)))


# ---------------------------------------------------------------- チューニングと学習
def tune(kind: str, transform: str, X: pd.DataFrame, y: pd.Series, cfg: dict
         ) -> tuple[dict, float]:
    """拡張窓 CV(元スケールの RMSE)でグリッドサーチし、最良パラメータと CV-RMSE を返す。"""
    grid = cfg["model"][kind]["grid"]
    keys = list(grid)
    best, best_score = None, np.inf
    for values in product(*(grid[k] for k in keys)):
        params = dict(zip(keys, values))
        scores = []
        for tr, va in cv_splitter(cfg["split"]["cv_splits"]).split(X):
            m = TSModel(kind, make_estimator(kind, params, cfg), transform, params)
            m.fit(X.iloc[tr], y.iloc[tr])
            scores.append(_rmse(y.iloc[va], m.predict(X.iloc[va])))
        if np.mean(scores) < best_score:
            best, best_score = params, float(np.mean(scores))
    return best, best_score


def fit_tuned(kind: str, transform: str, X_train: pd.DataFrame, y_train: pd.Series,
              cfg: dict) -> tuple[TSModel, float]:
    params, cv = tune(kind, transform, X_train, y_train, cfg)
    est = make_estimator(kind, params, cfg)
    return TSModel(kind, clone(est), transform, params).fit(X_train, y_train), cv


# ---------------------------------------------------------------- 比較表
def naive_predictions(X: pd.DataFrame, naive: dict[str, int]) -> dict[str, np.ndarray]:
    """Naive 予測 = k日前の実績(ラグ特徴量の列そのもの)。"""
    return {name: X[f"lag_{k}"].to_numpy() for name, k in naive.items()}


@produces_table
def accuracy_table(y_test: pd.Series, preds: dict[str, np.ndarray],
                   cv_rmse: dict[str, float] | None = None,
                   params: dict[str, dict] | None = None) -> pd.DataFrame:
    """テスト期間の RMSE・MAE・MAPE・sMAPE(Naive を含むモデル比較)。"""
    rows = []
    for name, p in preds.items():
        row = {"モデル": name, **metrics(y_test, p)}
        if cv_rmse is not None:
            row["CV-RMSE(訓練期間)"] = cv_rmse.get(name, np.nan)
        if params is not None:
            row["パラメータ"] = str(params.get(name, ""))
        rows.append(row)
    return pd.DataFrame(rows).set_index("モデル")
