"""What-if シミュレーション: モデル再予測による変化量と、ΔSHAP による分解。

SHAP値は「現在の値と平均との差による寄与」であり「値を動かしたときの変化量」ではない。
変化量の根拠は常に model.predict の再実行(A)とし、SHAP はその説明(B)に使う。

変更の指定は {特徴量: (種類, 量)}。種類は
    "sd"  : 頑健σ(robust_sd)の倍数だけ加算
    "abs" : 元の単位で加算
    "pct" : 現在値に対する割合だけ加算(0.10 = +10%)
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .explain import compute_shap
from .provenance import produces_table

Change = tuple[str, float]
Scenario = dict[str, Change]


def _bound_labels(lower_q: float, upper_q: float) -> list[str]:
    return [f"下限({lower_q:.0%})", f"上限({upper_q:.0%})"]


def feature_bounds(X_train: pd.DataFrame, lower_q: float = 0.01,
                   upper_q: float = 0.99) -> pd.DataFrame:
    """動かしてよい範囲 = 訓練データの [lower_q, upper_q] 分位点。"""
    return X_train.quantile([lower_q, upper_q]).T.set_axis(_bound_labels(lower_q, upper_q),
                                                           axis=1)


def robust_sd(X_train: pd.DataFrame, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    """[lower_q, upper_q] 分位点の範囲に収まるデータで計算した標準偏差(外れ値で膨らまない)。"""
    lo, hi = X_train.quantile(lower_q), X_train.quantile(upper_q)
    return X_train.apply(lambda c: c[(c >= lo[c.name]) & (c <= hi[c.name])].std())


@produces_table
def ranges_table(X_train: pd.DataFrame, lower_q: float = 0.01,
                 upper_q: float = 0.99) -> pd.DataFrame:
    """頑健σ・通常σ・動かせる範囲の一覧。"""
    return pd.concat([robust_sd(X_train, lower_q, upper_q).rename("頑健σ"),
                      X_train.std().rename("通常σ"),
                      feature_bounds(X_train, lower_q, upper_q)], axis=1)


def resolve_delta(kind: str, amount: float, current: float, sd: float) -> float:
    if kind == "sd":
        return amount * sd
    if kind == "abs":
        return amount
    if kind == "pct":
        return amount * current
    raise ValueError(kind)


def apply_changes(x: pd.Series, changes: Scenario, sd: pd.Series,
                  bounds: pd.DataFrame) -> tuple[pd.Series, list[dict]]:
    """変更を適用し、bounds の範囲にクリップする。範囲外の要求は警告として返す。"""
    new = x.copy()
    log = []
    for f, (kind, amount) in changes.items():
        requested = x[f] + resolve_delta(kind, amount, x[f], sd[f])
        lo, hi = bounds.loc[f].iloc[0], bounds.loc[f].iloc[1]
        applied = float(np.clip(requested, lo, hi))
        new[f] = applied
        log.append({"特徴量": f, "元の値": x[f], "要求値": requested, "適用値": applied,
                    "外挿警告": not np.isclose(requested, applied)})
    return new, log


def _predict_one(model, row: pd.Series) -> float:
    return float(model.predict(row.to_frame().T.astype(float))[0])


@produces_table
def single_variable_whatif(model, x: pd.Series, sd: pd.Series, bounds: pd.DataFrame,
                           kind: str = "sd", amount: float = 1.0,
                           exclude: Iterable[str] = (),
                           correlated_pairs: Iterable[tuple[str, str]] = ()) -> pd.DataFrame:
    """各特徴量を単独で動かしたときの Δ予測値(モデル再予測)の表。

    exclude: 動かさない特徴量。correlated_pairs: 片方だけ動かすと非現実的になりうるペア(注記用)。
    """
    exclude, pairs = set(exclude), list(correlated_pairs)
    base_pred = _predict_one(model, x)
    rows = []
    for f in x.index:
        if f in exclude:
            continue
        new, log = apply_changes(x, {f: (kind, amount)}, sd, bounds)
        new_pred = _predict_one(model, new)
        partners = [b if a == f else a for a, b in pairs if f in (a, b)]
        rows.append({**log[0], "変更量": log[0]["適用値"] - log[0]["元の値"],
                     "元の予測": base_pred, "変更後の予測": new_pred,
                     "Δ予測値": new_pred - base_pred,
                     "相関注記": f"{partners[0]}と強相関(片方のみ変更)" if partners else ""})
    df = pd.DataFrame(rows)
    return df.reindex(df["Δ予測値"].abs().sort_values(ascending=False).index,
                      ).reset_index(drop=True)


@produces_table
def scenario_analysis(model, explainer, x: pd.Series, sd: pd.Series, bounds: pd.DataFrame,
                      scenarios: dict[str, Scenario],
                      correlated_pairs: Iterable[tuple[str, str]] = ()
                      ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """複数変数を同時に動かすシナリオの Δ予測値・相互作用分・ΔSHAP 分解(summary / dshap / changes)。

    返り値:
      summary: シナリオ別の Δ予測値・単独Δの合計・相互作用分(非加法分)・ΣΔSHAP
      dshap:   シナリオ×特徴量 の ΔSHAP(行和 = Δ予測値)
      changes: 適用した変更と外挿警告
    """
    pairs = list(correlated_pairs)
    base_pred = _predict_one(model, x)
    sv_before = compute_shap(explainer, x.to_frame().T.astype(float)).values[0]
    summary, dshap_rows, change_rows = [], {}, []
    for name, changes in scenarios.items():
        new, log = apply_changes(x, changes, sd, bounds)
        joint = _predict_one(model, new) - base_pred
        # 単独で動かした Δ の合計(同じクリップ後の値を使う)
        singles = {}
        for entry in log:
            one = x.copy()
            one[entry["特徴量"]] = entry["適用値"]
            singles[entry["特徴量"]] = _predict_one(model, one) - base_pred
        sv_after = compute_shap(explainer, new.to_frame().T.astype(float)).values[0]
        d = sv_after - sv_before
        dshap_rows[name] = d
        corr_note = [f"{a}/{b}" for a, b in pairs if (a in changes) != (b in changes)]
        summary.append({"シナリオ": name, "元の予測": base_pred, "変更後の予測": base_pred + joint,
                        "Δ予測値(同時)": joint, "単独Δの合計": sum(singles.values()),
                        "相互作用分(同時−単独合計)": joint - sum(singles.values()),
                        "ΣΔSHAP": d.sum(),
                        "単独Δの内訳": ", ".join(f"{k}:{v:+.3f}" for k, v in singles.items()),
                        "外挿警告": any(e["外挿警告"] for e in log),
                        "相関注記": "片方のみ変更: " + ", ".join(corr_note) if corr_note else ""})
        change_rows += [{"シナリオ": name, **e} for e in log]
    dshap = pd.DataFrame(dshap_rows, index=x.index).T
    return pd.DataFrame(summary), dshap, pd.DataFrame(change_rows)
