"""SHAP 計算(元スケールへの変換)・グループ集約・加法性チェック・代表日の分解。

level 変換(モデルは cnt − roll7_mean を学習)では、元スケールの予測値は
    予測値 = roll7_mean + f(x) = [E[f] + 訓練平均roll7] + Σφ_j + (roll7_mean − 訓練平均roll7)
と分解できる。最後の項を「直近水準」という擬似特徴量として自己回帰(AR)グループに入れると、
    ベースライン(= E[f] + 訓練平均roll7) + Σ 寄与 = 予測値
が元の単位(件/日)でそのまま成り立つ。ベースラインは訓練期間の平均予測にほぼ等しい。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap

from .model import LEVEL_FEATURE, TSModel
from .xai_platform.explain import (DEFAULT_ADDITIVITY_TOL, baseline, linear_shap_check,
                                   make_tree_explainer)
from .xai_platform.provenance import produces_table
from .xai_platform.units import Unit

WEEKDAY_JA = ["日", "月", "火", "水", "木", "金", "土"]


def date_label(d: pd.Timestamp) -> str:
    return f"{d:%Y-%m-%d}({WEEKDAY_JA[(d.dayofweek + 1) % 7]})"


@dataclass
class OriginalScaleExplainer:
    """TSModel の予測を元スケールで「ベースライン + Σ寄与」に分解する。"""
    model: TSModel
    level_ref: float          # 訓練期間の roll7_mean の平均(level 変換のとき)
    level_name: str

    def __post_init__(self):
        if self.model.transform not in ("none", "level"):
            raise NotImplementedError("元スケールで加法になるのは none / level 変換のみ")
        self.tree = make_tree_explainer(self.model.estimator)

    @classmethod
    def from_train(cls, model: TSModel, X_train: pd.DataFrame, level_name: str):
        ref = float(X_train[LEVEL_FEATURE].mean()) if model.transform == "level" else 0.0
        return cls(model, ref, level_name)

    @property
    def base_raw(self) -> float:
        """モデルスケールのベースライン E[f]。"""
        return baseline(self.tree)

    @property
    def base(self) -> float:
        """元スケールのベースライン。"""
        return self.base_raw + self.level_ref

    def raw(self, X: pd.DataFrame) -> shap.Explanation:
        """モデルスケール(cnt − roll7_mean)の SHAP。"""
        return self.tree(X)

    def __call__(self, X: pd.DataFrame) -> shap.Explanation:
        """元スケールの SHAP(level 変換なら最終列に直近水準の擬似特徴量を追加)。"""
        sv = self.raw(X)
        values, data, names = sv.values, X.to_numpy(float), list(X.columns)
        if self.model.transform == "level":
            offset = (X[LEVEL_FEATURE] - self.level_ref).to_numpy()[:, None]
            values = np.hstack([values, offset])
            data = np.hstack([data, X[[LEVEL_FEATURE]].to_numpy(float)])
            names = names + [self.level_name]
        return shap.Explanation(values=values, base_values=np.full(len(X), self.base),
                                data=data, feature_names=names)

    def groups(self, groups: dict[str, list[str]], ar_group: str) -> dict[str, list[str]]:
        """直近水準の擬似特徴量を AR グループに加えたグループ定義。"""
        out = {g: list(fs) for g, fs in groups.items()}
        if self.model.transform == "level":
            out[ar_group].append(self.level_name)
        return out


def to_frame(sv: shap.Explanation, index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(sv.values, index=index, columns=sv.feature_names)


def group_frame(sv: shap.Explanation, index: pd.Index,
                groups: dict[str, list[str]]) -> pd.DataFrame:
    """日付 × グループの寄与。groups は全特徴量を過不足なく覆うこと。"""
    vals = to_frame(sv, index)
    covered = sorted(f for fs in groups.values() for f in fs)
    assert covered == sorted(vals.columns), "グループ定義が全特徴量を過不足なく覆っていない"
    return pd.DataFrame({g: vals[fs].sum(axis=1) for g, fs in groups.items()})


# ---------------------------------------------------------------- 加法性チェック
@produces_table
def additivity_table(ex: OriginalScaleExplainer, X: pd.DataFrame,
                     tol: float = DEFAULT_ADDITIVITY_TOL) -> pd.DataFrame:
    """加法性チェック(モデルスケール・元スケール)の最大絶対誤差と合否。"""
    raw = ex.raw(X)
    err_raw = np.abs(ex.base_raw + raw.values.sum(axis=1) - ex.model.predict_raw(X))
    sv = ex(X)
    err = np.abs(ex.base + sv.values.sum(axis=1) - ex.model.predict(X))
    return pd.DataFrame([
        {"スケール": "モデル(cnt − roll7_mean)", "式": "E[f] + Σφ = f(x)",
         "日数": len(X), "最大絶対誤差": err_raw.max(), "許容誤差": tol, "合格": bool((err_raw < tol).all())},
        {"スケール": "元(件/日)", "式": "ベースライン + Σ寄与 = 予測値",
         "日数": len(X), "最大絶対誤差": err.max(), "許容誤差": tol, "合格": bool((err < tol).all())},
    ]).set_index("スケール")


@produces_table
def ridge_check(ridge: TSModel, X_train: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
    """Ridge(標準化→Ridge)の SHAP が 係数 × (x − 訓練平均) と一致するかの検証表。

    標準化は特徴量ごとの1次変換なので、標準化後の特徴量で計算した SHAP は元の特徴量の SHAP と等しい。
    """
    scaler, lin = ridge.estimator[0], ridge.estimator[-1]
    Xs_tr = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
    Xs_te = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
    df = linear_shap_check(lin, Xs_tr, Xs_te)
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


def _fmt_value(f: str, v: float) -> str:
    if float(v).is_integer() or abs(v) >= 100:
        return f"{v:,.0f}"
    return f"{v:.3g}"


@produces_table
def group_decomposition_table(sv_row: shap.Explanation, groups: dict[str, list[str]],
                              unit: Unit = Unit(), top: int = 3,
                              pred: float | None = None, actual: float | None = None
                              ) -> pd.DataFrame:
    """1日の分解表: グループ | 主な特徴量と値 | 寄与合計。最終行が ベースライン + Σ = 予測値。"""
    s = pd.Series(sv_row.values, index=sv_row.feature_names)
    x = pd.Series(sv_row.data, index=sv_row.feature_names)
    base = float(np.ravel(sv_row.base_values)[0])
    rows = [{"グループ": "ベースライン(訓練期間の平均的な予測)", "主な特徴量と値(寄与)": "",
             "寄与合計": base}]
    for g, fs in sorted(groups.items(), key=lambda kv: -abs(s[kv[1]].sum())):
        main = s[fs].reindex(s[fs].abs().sort_values(ascending=False).index)[:top]
        rows.append({"グループ": g, "主な特徴量と値(寄与)": ", ".join(
            f"{f}={_fmt_value(f, x[f])}({unit.fmt(v, True)})" for f, v in main.items()),
            "寄与合計": s[fs].sum()})
    total = base + s.sum()
    rows.append({"グループ": "予測値 = ベースライン + Σ寄与", "主な特徴量と値(寄与)": "",
                 "寄与合計": total})
    if pred is not None:
        assert abs(pred - total) < 1e-6, "分解表の合計が予測値と一致しない"
    if actual is not None:
        rows.append({"グループ": "(参考)実測値", "主な特徴量と値(寄与)": "", "寄与合計": actual})
    return pd.DataFrame(rows)


def summary_sentence(day: pd.Timestamp, sv_row: shap.Explanation,
                     groups: dict[str, list[str]], unit: Unit,
                     actual: float | None = None) -> str:
    """「ベースライン○に対し、ARが+○(主に…)、…で、予測は○」の一文。"""
    s = pd.Series(sv_row.values, index=sv_row.feature_names)
    x = pd.Series(sv_row.data, index=sv_row.feature_names)
    base = float(np.ravel(sv_row.base_values)[0])
    parts = []
    for g, fs in sorted(groups.items(), key=lambda kv: -abs(s[kv[1]].sum())):
        tot = s[fs].sum()
        f = s[fs].abs().idxmax()
        val = "" if f.startswith("直近水準") else f"={_fmt_value(f, x[f])}"
        parts.append(f"{g}が{unit.fmt(tot, True)}(主に{f}{val}で{unit.fmt(s[f], True)})")
    tail = f"。実測は{unit.fmt(actual)}{unit.label}" if actual is not None else ""
    return (f"{date_label(day)}: ベースライン{unit.fmt(base)}に対し、" + "、".join(parts)
            + f"で、予測は{unit.fmt(base + s.sum())}{unit.label}{tail}。")
