"""このデータ固有の説明処理: Facade(ShapAnalyzer)の組み立て・代表日の選定・Ridge での検証。

level 変換(モデルは cnt − roll7_mean を学習)では、予測値 = モデル出力 + roll7_mean なので、
roll7_mean を ShapAnalyzer の補正項(Offset)として渡す。Offset は
    「訓練期間の roll7_mean 平均」→ ベースラインへ
    「roll7_mean − 訓練平均」      → 擬似特徴量「直近水準」の寄与として AR グループへ
と分けられ、ベースライン(= E[f] + 訓練平均roll7)+ Σ寄与 = 予測値 が件/日のまま成り立つ。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from .model import LEVEL_FEATURE, TSModel
from .xai_platform import Offset, ShapAnalyzer, Unit, produces_table

WEEKDAY_JA = ["日", "月", "火", "水", "木", "金", "土"]


def date_label(d: pd.Timestamp) -> str:
    return f"{d:%Y-%m-%d}({WEEKDAY_JA[(d.dayofweek + 1) % 7]})"


def make_analyzer(model: TSModel, X_train: pd.DataFrame, cfg: dict, groups: dict[str, list[str]],
                  unit: Unit, out_dir=None) -> ShapAnalyzer:
    """TSModel を ShapAnalyzer に渡す。none / level 変換のみ(log は元スケールで加法にならない)。"""
    if model.transform not in ("none", "level"):
        raise NotImplementedError("元スケールで加法になるのは none / level 変換のみ")
    offset = None
    if model.transform == "level":
        offset = Offset(name=cfg["level_name"], func=lambda X: X[LEVEL_FEATURE],
                        features=(LEVEL_FEATURE,), group=cfg["groups"]["ar"])
    wc = cfg["whatif"]
    return ShapAnalyzer(model.estimator, X_train, unit=unit, groups=groups, offset=offset,
                        out_dir=out_dir, group_colors=cfg["group_colors"],
                        bounds_quantiles=tuple(wc["bounds_quantiles"]))


# ---------------------------------------------------------------- Ridge での検証
@produces_table
def ridge_check(ridge: TSModel, X_train: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
    """Ridge(標準化→Ridge)の SHAP が 係数 × (x − 訓練平均) と一致するかの検証表。

    標準化は特徴量ごとの1次変換なので、標準化後の特徴量で計算した SHAP は元の特徴量の SHAP と等しい。
    """
    scaler, lin = ridge.estimator[0], ridge.estimator[-1]
    Xs_tr = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
    Xs_te = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
    df = ShapAnalyzer.linear_check(lin, Xs_tr, Xs_te)
    df.insert(1, "元の単位の係数(1単位あたり)", lin.coef_ / scaler.scale_)
    return df


def ridge_explanation(ridge: TSModel, X_train: pd.DataFrame, X: pd.DataFrame,
                      level_ref: float, level_name: str) -> tuple[shap.Explanation, float]:
    """Ridge の元スケール寄与(係数 × (x − 訓練平均)+ 直近水準)とベースライン。"""
    scaler, lin = ridge.estimator[0], ridge.estimator[-1]
    coef = lin.coef_ / scaler.scale_
    mu = X_train.mean().to_numpy()
    vals = (X.to_numpy(float) - mu) * coef
    base = float(lin.intercept_ + (scaler.transform(X_train).mean(axis=0) * lin.coef_).sum())
    if ridge.transform == "level":
        vals = np.hstack([vals, (X[LEVEL_FEATURE] - level_ref).to_numpy()[:, None]])
        base += level_ref
    names = list(X.columns) + ([level_name] if ridge.transform == "level" else [])
    return shap.Explanation(values=vals, base_values=np.full(len(X), base),
                            data=np.zeros_like(vals), feature_names=names), base


# ---------------------------------------------------------------- 代表日
@produces_table
def select_days(pred_all: pd.Series, y: pd.Series, test_index: pd.Index,
                criteria: dict[str, str]) -> pd.DataFrame:
    """代表日の選定(テスト期間内)。同じ日が重なったら次点を採る。

    pred_max / pred_min: 予測値の最大 / 最小
    diff_max / diff_min: 前日の予測値からの変化の最大(上昇)/ 最小(下落)
    error_max:           |実測 − 予測| の最大
    """
    p = pred_all.loc[test_index]
    diff = pred_all.diff().loc[test_index]
    err = (y.loc[test_index] - p).abs()
    scores = {"pred_max": p, "pred_min": -p, "diff_max": diff, "diff_min": -diff,
              "error_max": err}
    basis = {"pred_max": "予測値", "pred_min": "予測値", "diff_max": "前日比Δ予測値",
             "diff_min": "前日比Δ予測値", "error_max": "|実測−予測|"}
    rows, used = [], set()
    for label, kind in criteria.items():
        s = scores[kind].dropna().sort_values(ascending=False)
        d = next(d for d in s.index if d not in used)
        used.add(d)
        rows.append({"区分": label, "日付": d, "曜日付き": date_label(d), "選定基準": basis[kind],
                     "基準値": {"pred_min": p[d], "diff_min": diff[d]}.get(kind, s[d]),
                     "予測値": p[d], "実測値": y[d], "前日比Δ予測値": diff[d]})
    return pd.DataFrame(rows)
