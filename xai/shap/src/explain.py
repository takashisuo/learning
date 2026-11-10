"""SHAP計算・加法性チェック・予測値の分解表。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from .data import SEED, TARGET_UNIT

ADDITIVITY_TOL = 1e-6  # |expected_value + Σshap − predict| の許容誤差(目的変数の単位)

# 全体構成の積み上げ図で使う特徴量グループ
FEATURE_GROUPS = {
    "所得": ["MedInc"],
    "立地": ["Latitude", "Longitude"],
    "住居": ["AveRooms", "AveBedrms", "AveOccup"],
    "築年": ["HouseAge"],
    "人口": ["Population"],
}


# ---------------------------------------------------------------- 表記
def to_man_dollar(v: float) -> float:
    """目的変数(10万ドル単位)を万ドルに換算する。"""
    return v * 10.0


def fmt_signed(v: float, digits: int = 3) -> str:
    """'+0.820(約8.2万ドル)' のような符号付き・単位付き表記。"""
    sign = "+" if v >= 0 else "−"
    return f"{sign}{abs(v):.{digits}f}(約{sign}{abs(to_man_dollar(v)):.1f}万ドル)"


def fmt_level(v: float, digits: int = 3) -> str:
    """'2.950(約29.5万ドル)' のような水準の表記。"""
    return f"{v:.{digits}f}(約{to_man_dollar(v):.1f}万ドル)"


# ---------------------------------------------------------------- SHAP計算
def make_tree_explainer(model) -> shap.TreeExplainer:
    # tree_path_dependent: ベースライン=訓練データの平均予測。相互作用値もこの方式でのみ計算可能。
    return shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")


def baseline(explainer) -> float:
    return float(np.ravel(explainer.expected_value)[0])


def compute_shap(explainer, X: pd.DataFrame) -> shap.Explanation:
    return explainer(X)


def additivity_check(explainer, model, X: pd.DataFrame, sv: shap.Explanation,
                     tol: float = ADDITIVITY_TOL) -> pd.DataFrame:
    """各サンプルの |ベースライン + ΣSHAP − predict| を返す。"""
    recon = baseline(explainer) + sv.values.sum(axis=1)
    pred = model.predict(X)
    err = np.abs(recon - pred)
    return pd.DataFrame({"予測値": pred, "再構成値": recon, "絶対誤差": err,
                         "合格": err < tol}, index=X.index)


def additivity_summary(check: pd.DataFrame, tol: float = ADDITIVITY_TOL) -> dict:
    return {"サンプル数": len(check), "許容誤差": tol,
            "最大絶対誤差": float(check["絶対誤差"].max()),
            "合格数": int(check["合格"].sum()), "全件合格": bool(check["合格"].all())}


def ridge_shap_check(ridge, X_train: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
    """線形モデルでは SHAP = 係数 × (x − 訓練平均)。shap.LinearExplainer の値と比較する。"""
    # DataFrame をそのまま渡すと背景データが100件にサブサンプルされ平均がずれるため、全件を明示する
    masker = shap.maskers.Independent(X_train, max_samples=len(X_train))
    explainer = shap.LinearExplainer(ridge, masker)
    sv = explainer(X_test).values
    manual = (X_test.values - X_train.values.mean(axis=0)) * ridge.coef_
    diff = np.abs(sv - manual).max(axis=0)
    base_err = abs(baseline(explainer) - ridge.predict(X_train).mean())
    df = pd.DataFrame({"係数": ridge.coef_, "最大絶対差(SHAP vs 係数×(x−平均))": diff},
                      index=X_test.columns)
    df.attrs["ベースライン誤差"] = base_err
    return df


def interaction_values(explainer, X: pd.DataFrame, n: int = 1000,
                       seed: int = SEED) -> tuple[np.ndarray, pd.DataFrame]:
    """相互作用値(計算が重いので n 件にサブサンプル)。"""
    if len(X) > n:
        idx = np.random.default_rng(seed).choice(len(X), size=n, replace=False)
        X = X.iloc[np.sort(idx)]
    return explainer.shap_interaction_values(X), X


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


def main_vs_interaction(inter: np.ndarray, features: list[str]) -> pd.DataFrame:
    """各特徴量のSHAP値を主効果(対角)と相互作用分(非対角の和)に分けた平均絶対値。"""
    main = np.abs(np.diagonal(inter, axis1=1, axis2=2)).mean(axis=0)
    total = inter.sum(axis=2)
    inter_part = total - np.diagonal(inter, axis1=1, axis2=2)
    return pd.DataFrame({"平均|SHAP|": np.abs(total).mean(axis=0),
                         "平均|主効果|": main,
                         "平均|相互作用分|": np.abs(inter_part).mean(axis=0)},
                        index=features).sort_values("平均|SHAP|", ascending=False)


# ---------------------------------------------------------------- 代表サンプルと分解表
REPRESENTATIVE_QUANTILES = {"予測が高い(95%点)": 0.95, "中央値付近(50%点)": 0.50,
                            "予測が低い(5%点)": 0.05}


def select_representatives(pred: pd.Series) -> pd.DataFrame:
    """予測値の分位点に最も近いテストサンプルを選ぶ。"""
    rows = []
    for label, q in REPRESENTATIVE_QUANTILES.items():
        target = pred.quantile(q)
        idx = (pred - target).abs().idxmin()
        rows.append({"区分": label, "分位点": q, "分位点の予測値": target,
                     "サンプルindex": idx, "予測値": pred[idx]})
    return pd.DataFrame(rows)


def decomposition_table(sv_row: shap.Explanation, base: float) -> pd.DataFrame:
    """特徴量 | 現在の値 | SHAP値 | 累積。|SHAP|の大きい順に積み上げる。"""
    df = pd.DataFrame({"特徴量": sv_row.feature_names, "現在の値": sv_row.data,
                       "SHAP値": sv_row.values})
    df = df.reindex(df["SHAP値"].abs().sort_values(ascending=False).index)
    df["SHAP値(万ドル)"] = to_man_dollar(df["SHAP値"])
    df["累積"] = base + df["SHAP値"].cumsum()
    head = pd.DataFrame([{"特徴量": "ベースライン", "現在の値": np.nan, "SHAP値": np.nan,
                          "SHAP値(万ドル)": np.nan, "累積": base}])
    total = base + sv_row.values.sum()
    tail = pd.DataFrame([{"特徴量": "予測値 = ベースライン + ΣSHAP", "現在の値": np.nan,
                          "SHAP値": sv_row.values.sum(),
                          "SHAP値(万ドル)": to_man_dollar(sv_row.values.sum()),
                          "累積": total}])
    return pd.concat([head, df, tail], ignore_index=True)


def summary_sentence(sv_row: shap.Explanation, base: float, top: int = 3) -> str:
    """日本語の一文要約。"""
    order = np.argsort(-np.abs(sv_row.values))
    parts = [f"{sv_row.feature_names[i]}(={sv_row.data[i]:.3g})が{fmt_signed(sv_row.values[i])}"
             for i in order[:top]]
    rest = sv_row.values[order[top:]].sum()
    pred = base + sv_row.values.sum()
    return (f"ベースライン{fmt_level(base)}に対し、" + "、".join(parts)
            + f"、その他{len(order) - top}特徴量の合計が{fmt_signed(rest)}で、"
            f"予測は{fmt_level(pred)}。")


# ---------------------------------------------------------------- 全体構成
def group_contributions(sv: shap.Explanation, groups: dict = FEATURE_GROUPS) -> pd.DataFrame:
    names = list(sv.feature_names)
    covered = [f for fs in groups.values() for f in fs]
    assert sorted(covered) == sorted(names), "グループ定義が全特徴量を過不足なく覆っていない"
    vals = pd.DataFrame(sv.values, columns=names)
    return pd.DataFrame({g: vals[fs].sum(axis=1) for g, fs in groups.items()})


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


def dependence_table(sv: shap.Explanation, feature: str, n_bins: int = 10) -> pd.DataFrame:
    """特徴量の値の分位ビンごとの平均SHAP値(「この範囲だと予測がいくら上下するか」)。"""
    j = list(sv.feature_names).index(feature)
    x, s = sv.data[:, j], sv.values[:, j]
    bins = pd.qcut(x, n_bins, duplicates="drop")
    df = pd.DataFrame({"値の範囲": bins, "SHAP": s})
    out = df.groupby("値の範囲", observed=True)["SHAP"].agg(["mean", "count"])
    out.columns = ["平均SHAP値", "件数"]
    out["平均SHAP値(万ドル)"] = to_man_dollar(out["平均SHAP値"])
    out = out.reset_index()
    out["値の範囲"] = [f"{iv.left:.4g} – {iv.right:.4g}" for iv in out["値の範囲"]]
    return out


def unit_note() -> str:
    return f"数値の単位は目的変数(MedHouseVal)の単位={TARGET_UNIT}。1.0 = 10万ドル。"
