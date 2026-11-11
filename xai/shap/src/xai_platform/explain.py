"""SHAP計算・加法性チェック・予測値の分解表・相互作用・グループ集計。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from .provenance import produces_table
from .units import Unit

DEFAULT_ADDITIVITY_TOL = 1e-6  # |expected_value + Σshap − predict| の許容誤差(目的変数の単位)


# ---------------------------------------------------------------- SHAP計算
def make_tree_explainer(model) -> shap.TreeExplainer:
    """木モデル用 Explainer。

    tree_path_dependent: ベースライン = 訓練データの平均予測(行サンプリングなしの場合)。
    相互作用値もこの方式でのみ計算できる。
    """
    return shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")


def baseline(explainer) -> float:
    return float(np.ravel(explainer.expected_value)[0])


def compute_shap(explainer, X: pd.DataFrame) -> shap.Explanation:
    return explainer(X)


@produces_table
def additivity_check(explainer, model, X: pd.DataFrame, sv: shap.Explanation,
                     tol: float = DEFAULT_ADDITIVITY_TOL) -> pd.DataFrame:
    """加法性チェック: 各サンプルの |ベースライン + ΣSHAP − predict| と合否の表。"""
    recon = baseline(explainer) + sv.values.sum(axis=1)
    pred = model.predict(X)
    err = np.abs(recon - pred)
    return pd.DataFrame({"予測値": pred, "再構成値": recon, "絶対誤差": err,
                         "合格": err < tol}, index=X.index)


def additivity_summary(check: pd.DataFrame, tol: float = DEFAULT_ADDITIVITY_TOL) -> dict:
    return {"サンプル数": len(check), "許容誤差": tol,
            "最大絶対誤差": float(check["絶対誤差"].max()),
            "合格数": int(check["合格"].sum()), "全件合格": bool(check["合格"].all())}


@produces_table
def linear_shap_check(linear_model, X_train: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
    """線形モデルのSHAP値が 係数 × (x − 訓練平均) と一致するかの検証表。

    coef_ を持つ線形モデルが対象。shap.LinearExplainer の値と手計算を比較し、SHAP 実装の正しさを検証する。
    結果の attrs["ベースライン誤差"] に |expected_value − 訓練データの平均予測| を入れる。
    """
    # DataFrame をそのまま渡すと背景データが100件にサブサンプルされ平均がずれるため、全件を明示する
    masker = shap.maskers.Independent(X_train, max_samples=len(X_train))
    explainer = shap.LinearExplainer(linear_model, masker)
    sv = explainer(X_test).values
    manual = (X_test.values - X_train.values.mean(axis=0)) * linear_model.coef_
    diff = np.abs(sv - manual).max(axis=0)
    base_err = abs(baseline(explainer) - linear_model.predict(X_train).mean())
    df = pd.DataFrame({"係数": linear_model.coef_, "最大絶対差(SHAP vs 係数×(x−平均))": diff},
                      index=X_test.columns)
    df.attrs["ベースライン誤差"] = base_err
    return df


@produces_table
def mean_abs_shap(sv: shap.Explanation) -> pd.Series:
    """特徴量ごとの平均|SHAP|(大きい順)。"""
    return (pd.Series(np.abs(sv.values).mean(axis=0), index=sv.feature_names)
            .sort_values(ascending=False))


# ---------------------------------------------------------------- 相互作用
def interaction_values(explainer, X: pd.DataFrame, n: int | None = None,
                       seed: int = 0) -> tuple[np.ndarray, pd.DataFrame]:
    """相互作用値。計算が重いので n 件にサブサンプルできる(n=None なら全件)。"""
    if n is not None and len(X) > n:
        idx = np.random.default_rng(seed).choice(len(X), size=n, replace=False)
        X = X.iloc[np.sort(idx)]
    return explainer.shap_interaction_values(X), X


@produces_table
def interaction_strength(inter: np.ndarray, features: list[str]) -> pd.DataFrame:
    """ペアごとの相互作用の強さ = 平均|φ_ij + φ_ji|(φ_ij は対称に半分ずつ配分されている)。"""
    rows = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            pair = inter[:, i, j] + inter[:, j, i]
            rows.append({"特徴量1": features[i], "特徴量2": features[j],
                         "平均|相互作用|": float(np.abs(pair).mean())})
    return pd.DataFrame(rows).sort_values("平均|相互作用|", ascending=False,
                                          ignore_index=True)


@produces_table
def main_vs_interaction(inter: np.ndarray, features: list[str]) -> pd.DataFrame:
    """各特徴量のSHAP値を主効果(対角)と相互作用分(非対角の和)に分けた平均絶対値。"""
    main = np.abs(np.diagonal(inter, axis1=1, axis2=2)).mean(axis=0)
    total = inter.sum(axis=2)
    inter_part = total - np.diagonal(inter, axis1=1, axis2=2)
    return pd.DataFrame({"平均|SHAP|": np.abs(total).mean(axis=0),
                         "平均|主効果|": main,
                         "平均|相互作用分|": np.abs(inter_part).mean(axis=0)},
                        index=features).sort_values("平均|SHAP|", ascending=False)


def strongest_partner(sv: shap.Explanation, feature: str, X: pd.DataFrame) -> str:
    """依存プロットの色に使う、相互作用の強い特徴量(shap の近似による)。"""
    j = list(sv.feature_names).index(feature)
    return sv.feature_names[shap.utils.approximate_interactions(j, sv.values, X)[0]]


# ---------------------------------------------------------------- 代表サンプルと分解表
@produces_table
def select_representatives(pred: pd.Series, quantiles: dict[str, float]) -> pd.DataFrame:
    """予測値の分位点に最も近いサンプルを選ぶ。quantiles は {区分ラベル: 分位点}。"""
    rows = []
    for label, q in quantiles.items():
        target = pred.quantile(q)
        idx = (pred - target).abs().idxmin()
        rows.append({"区分": label, "分位点": q, "分位点の予測値": target,
                     "サンプルindex": idx, "予測値": pred[idx]})
    return pd.DataFrame(rows)


@produces_table
def decomposition_table(sv_row: shap.Explanation, base: float,
                        unit: Unit = Unit()) -> pd.DataFrame:
    """1サンプルの分解表(現在の値・SHAP値・累積。最終行が ベースライン + ΣSHAP = 予測値)。

    列: 特徴量 / 現在の値 / SHAP値 / (換算単位の列) / 累積。|SHAP|の大きい順に積み上げる。
    """
    alt = unit.alt_column("SHAP値")
    df = pd.DataFrame({"特徴量": sv_row.feature_names, "現在の値": sv_row.data,
                       "SHAP値": sv_row.values})
    df = df.reindex(df["SHAP値"].abs().sort_values(ascending=False).index)
    if alt:
        df[alt] = unit.to_alt(df["SHAP値"])
    df["累積"] = base + df["SHAP値"].cumsum()
    total = sv_row.values.sum()
    head = {"特徴量": "ベースライン", "現在の値": np.nan, "SHAP値": np.nan, "累積": base}
    tail = {"特徴量": "予測値 = ベースライン + ΣSHAP", "現在の値": np.nan, "SHAP値": total,
            "累積": base + total}
    if alt:
        head[alt], tail[alt] = np.nan, unit.to_alt(total)
    return pd.concat([pd.DataFrame([head]), df, pd.DataFrame([tail])],
                     ignore_index=True)[df.columns]


def summary_sentence(sv_row: shap.Explanation, base: float, unit: Unit = Unit(),
                     top: int = 3) -> str:
    """1サンプルの予測値の構成を日本語の一文で要約する(寄与の大きい上位 top 特徴量 + その他)。"""
    order = np.argsort(-np.abs(sv_row.values))
    parts = [f"{sv_row.feature_names[i]}(={sv_row.data[i]:.3g})が{unit.signed(sv_row.values[i])}"
             for i in order[:top]]
    rest = sv_row.values[order[top:]].sum()
    pred = base + sv_row.values.sum()
    return (f"ベースライン{unit.level(base)}に対し、" + "、".join(parts)
            + f"、その他{len(order) - top}特徴量の合計が{unit.signed(rest)}で、"
            f"予測は{unit.level(pred)}。")


# ---------------------------------------------------------------- 全体構成
@produces_table
def group_contributions(sv: shap.Explanation, groups: dict[str, list[str]]) -> pd.DataFrame:
    """特徴量グループ別にSHAP値を合算する。groups は全特徴量を過不足なく覆うこと。"""
    names = list(sv.feature_names)
    covered = [f for fs in groups.values() for f in fs]
    assert sorted(covered) == sorted(names), "グループ定義が全特徴量を過不足なく覆っていない"
    vals = pd.DataFrame(sv.values, columns=names)
    return pd.DataFrame({g: vals[fs].sum(axis=1) for g, fs in groups.items()})


@produces_table
def group_by_prediction_bins(group_df: pd.DataFrame, pred: np.ndarray,
                             n_bins: int = 10) -> pd.DataFrame:
    """予測値の分位ビンごとの平均グループ寄与。"""
    bins = pd.qcut(pred, n_bins, labels=False)
    df = group_df.copy()
    df["予測値"] = pred
    df["ビン"] = bins
    out = df.groupby("ビン").mean()
    out["件数"] = df.groupby("ビン").size()
    out["予測値の範囲"] = df.groupby("ビン")["予測値"].agg(
        lambda s: f"{s.min():.2f}–{s.max():.2f}")
    return out


@produces_table
def dependence_table(sv: shap.Explanation, feature: str, n_bins: int = 10,
                     unit: Unit = Unit()) -> pd.DataFrame:
    """特徴量の値の分位ビンごとの平均SHAP値(「この範囲だと予測がいくら上下するか」)。"""
    j = list(sv.feature_names).index(feature)
    x, s = sv.data[:, j], sv.values[:, j]
    bins = pd.qcut(x, n_bins, duplicates="drop")
    df = pd.DataFrame({"値の範囲": bins, "SHAP": s})
    out = df.groupby("値の範囲", observed=True)["SHAP"].agg(["mean", "count"])
    out.columns = ["平均SHAP値", "件数"]
    alt = unit.alt_column("平均SHAP値")
    if alt:
        out[alt] = unit.to_alt(out["平均SHAP値"])
    out = out.reset_index()
    out["値の範囲"] = [f"{iv.left:.4g} – {iv.right:.4g}" for iv in out["値の範囲"]]
    return out
