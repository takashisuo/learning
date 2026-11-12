"""グループ単位の分解: 寄与の集約・分解表・要約文・2時点の差(ΔSHAP)・行の並びに沿った寄与の表。"""
from __future__ import annotations

from typing import Hashable, Iterable

import numpy as np
import pandas as pd
import shap

from .provenance import produces_table
from .units import Unit


def group_frame(sv: shap.Explanation, index: pd.Index,
                groups: dict[str, list[str]]) -> pd.DataFrame:
    """行 × グループの寄与。groups は全特徴量を過不足なく覆うこと。"""
    vals = pd.DataFrame(sv.values, index=index, columns=sv.feature_names)
    covered = sorted(f for fs in groups.values() for f in fs)
    assert covered == sorted(vals.columns), "グループ定義が全特徴量を過不足なく覆っていない"
    return pd.DataFrame({g: vals[fs].sum(axis=1) for g, fs in groups.items()})


def _fmt_value(v: float) -> str:
    if float(v).is_integer() or abs(v) >= 100:
        return f"{v:,.0f}"
    return f"{v:.3g}"


@produces_table
def group_decomposition_table(sv_row: shap.Explanation, groups: dict[str, list[str]],
                              unit: Unit = Unit(), top: int = 3,
                              actual: float | None = None,
                              base_label: str = "ベースライン(訓練データの平均的な予測)"
                              ) -> pd.DataFrame:
    """1行の分解表: グループ | 主な特徴量と値(寄与) | 寄与合計。最終行が ベースライン + Σ = 予測値。"""
    s = pd.Series(sv_row.values, index=sv_row.feature_names)
    x = pd.Series(sv_row.data, index=sv_row.feature_names)
    base = float(np.ravel(sv_row.base_values)[0])
    rows = [{"グループ": base_label, "主な特徴量と値(寄与)": "", "寄与合計": base}]
    for g, fs in sorted(groups.items(), key=lambda kv: -abs(s[kv[1]].sum())):
        main = s[fs].reindex(s[fs].abs().sort_values(ascending=False).index)[:top]
        rows.append({"グループ": g, "主な特徴量と値(寄与)": ", ".join(
            f"{f}={_fmt_value(x[f])}({unit.fmt(v, True)})" for f, v in main.items()),
            "寄与合計": s[fs].sum()})
    rows.append({"グループ": "予測値 = ベースライン + Σ寄与", "主な特徴量と値(寄与)": "",
                 "寄与合計": base + s.sum()})
    if actual is not None:
        rows.append({"グループ": "(参考)実測値", "主な特徴量と値(寄与)": "", "寄与合計": actual})
    return pd.DataFrame(rows)


def summary_sentence(sv_row: shap.Explanation, groups: dict[str, list[str]], unit: Unit,
                     label: str = "", actual: float | None = None,
                     hide_value: Iterable[str] = ()) -> str:
    """「ベースライン○に対し、Aが+○(主に…)、…で、予測は○」の一文(グループ単位)。

    hide_value: 値を表示しない特徴量(補正項など、値そのものに意味が薄いもの)。
    """
    hide = set(hide_value)
    s = pd.Series(sv_row.values, index=sv_row.feature_names)
    x = pd.Series(sv_row.data, index=sv_row.feature_names)
    base = float(np.ravel(sv_row.base_values)[0])
    parts = []
    for g, fs in sorted(groups.items(), key=lambda kv: -abs(s[kv[1]].sum())):
        f = s[fs].abs().idxmax()
        val = "" if f in hide else f"={_fmt_value(x[f])}"
        parts.append(f"{g}が{unit.fmt(s[fs].sum(), True)}(主に{f}{val}で{unit.fmt(s[f], True)})")
    head = f"{label}: " if label else ""
    tail = f"。実測は{unit.fmt(actual)}{unit.label}" if actual is not None else ""
    return (f"{head}ベースライン{unit.fmt(base)}に対し、" + "、".join(parts)
            + f"で、予測は{unit.fmt(base + s.sum())}{unit.label}{tail}。")


@produces_table
def contribution_table(group_df: pd.DataFrame, base: float, pred: pd.Series,
                       actual: pd.Series | None = None, tol: float = 1e-6) -> pd.DataFrame:
    """行ごとの ベースライン + グループ別寄与 = 予測値(と実測値)。"""
    out = group_df.copy()
    out.insert(0, "ベースライン", base)
    out["予測値"] = pred
    if actual is not None:
        out["実測値"] = actual.reindex(out.index)
    assert np.allclose(out["ベースライン"] + group_df.sum(axis=1), out["予測値"], atol=tol)
    return out


@produces_table
def compare_table(contrib: pd.DataFrame, groups: list[str],
                  pairs: Iterable[tuple[Hashable, Hashable]]) -> pd.DataFrame:
    """2行の差の分解: 予測値(t) − 予測値(s) = Σ_グループ[寄与(t) − 寄与(s)]。ベースラインは差で消える。

    contrib は contribution_table の出力。pairs は (t, s) の組。s が無い組は飛ばす。
    """
    rows = []
    for t, s in pairs:
        if s not in contrib.index or t not in contrib.index:
            continue
        d = contrib.loc[t, groups] - contrib.loc[s, groups]
        row = {"対象": t, "比較対象": s, "予測値(比較対象)": contrib.loc[s, "予測値"],
               "予測値(対象)": contrib.loc[t, "予測値"],
               "Δ予測値": contrib.loc[t, "予測値"] - contrib.loc[s, "予測値"],
               **{f"Δ{g}": v for g, v in d.items()}, "ΣΔ寄与": d.sum()}
        if "実測値" in contrib.columns:
            row["Δ実測値"] = contrib.loc[t, "実測値"] - contrib.loc[s, "実測値"]
        rows.append(row)
    out = pd.DataFrame(rows).set_index("対象")
    out["恒等式の誤差"] = (out["Δ予測値"] - out["ΣΔ寄与"]).abs()
    return out


@produces_table
def profile_means(group_df: pd.DataFrame, keys, name: str) -> pd.DataFrame:
    """キー(季節・曜日・区分など)ごとのグループ別平均寄与と件数。"""
    out = group_df.groupby(keys, observed=True).mean()
    out["件数"] = group_df.groupby(keys, observed=True).size()
    out.index.name = name
    return out
