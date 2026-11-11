"""時間方向の分解(グループ別寄与の時系列・季節/曜日プロファイル・STL 比較)と、
前日比/前週比/前年同期比の変化の要因分解(ΔSHAP)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import MSTL

from .explain import WEEKDAY_JA, date_label
from .xai_platform.provenance import produces_table

SEASON_JA = {12: "冬", 1: "冬", 2: "冬", 3: "春", 4: "春", 5: "春",
             6: "夏", 7: "夏", 8: "夏", 9: "秋", 10: "秋", 11: "秋"}


# ---------------------------------------------------------------- 時間方向の構成
@produces_table
def contribution_timeseries(group_df: pd.DataFrame, base: float, pred: pd.Series,
                            y: pd.Series) -> pd.DataFrame:
    """日付ごとの ベースライン + グループ別寄与 = 予測値(と実測値)。"""
    out = group_df.copy()
    out.insert(0, "ベースライン", base)
    out["予測値"] = pred
    out["実測値"] = y.reindex(out.index)
    assert np.allclose(out["ベースライン"] + group_df.sum(axis=1), out["予測値"], atol=1e-6)
    return out


@produces_table
def profile_table(group_df: pd.DataFrame, by: str) -> pd.DataFrame:
    """季節・月・曜日ごとのグループ別平均寄与。by は season / month / weekday。"""
    idx = group_df.index
    if by == "season":
        key = pd.Categorical(idx.month.map(SEASON_JA), ["春", "夏", "秋", "冬"])
    elif by == "month":
        key = pd.Index(idx.month, name="月")
    elif by == "weekday":
        key = pd.Categorical([WEEKDAY_JA[(d + 1) % 7] for d in idx.dayofweek], WEEKDAY_JA)
    else:
        raise ValueError(by)
    out = group_df.groupby(key, observed=True).mean()
    out["日数"] = group_df.groupby(key, observed=True).size()
    out.index.name = {"season": "季節", "month": "月", "weekday": "曜日"}[by]
    return out


# ---------------------------------------------------------------- STL との比較
def stl_components(y: pd.Series, periods=(7, 365)) -> pd.DataFrame:
    """MSTL(週周期・年周期)の成分。2年分のデータなので年周期成分は粗い推定である点に注意。"""
    res = MSTL(y.astype(float), periods=periods).fit()
    seas = res.seasonal
    return pd.DataFrame({"トレンド": res.trend, "週周期": seas.iloc[:, 0],
                         "年周期": seas.iloc[:, 1], "残差": res.resid}, index=y.index)


@produces_table
def stl_comparison(stl: pd.DataFrame, feat_df: pd.DataFrame, group_df: pd.DataFrame,
                   parts: dict[str, list[str]], ar_group: str, exog_group: str,
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """STL の週周期・年周期成分と、モデルのカレンダー寄与(週内訳・年内訳)の比較表(曜日別・月別)。

    年周期は、この系列では気温(外生変数)と過去水準(AR)にも現れるので、それらの月別平均も並べる。
    いずれも各期間内での平均を引いた「平均からのずれ」で比較する。
    """
    idx = feat_df.index
    wk = pd.DataFrame({"STL 週周期成分": stl.loc[idx, "週周期"],
                       "カレンダー寄与(週内訳)": feat_df[parts["weekly"]].sum(axis=1)})
    wk_key = pd.Categorical([WEEKDAY_JA[(d + 1) % 7] for d in idx.dayofweek], WEEKDAY_JA)
    weekly = wk.groupby(wk_key, observed=True).mean()
    weekly = weekly - weekly.mean()
    weekly.index.name = "曜日"

    an = pd.DataFrame({"STL 年周期成分": stl.loc[idx, "年周期"],
                       "カレンダー寄与(年内訳)": feat_df[parts["annual"]].sum(axis=1),
                       f"{exog_group}寄与": group_df[exog_group],
                       f"{ar_group}寄与": group_df[ar_group]})
    annual = an.groupby(idx.month).mean()
    annual = annual - annual.mean()
    annual.index.name = "月"
    return weekly, annual


# ---------------------------------------------------------------- 変化の要因分解
@produces_table
def delta_decomposition(ts: pd.DataFrame, groups: list[str], lag: int,
                        dates: pd.Index) -> pd.DataFrame:
    """予測値(t) − 予測値(t−lag) = Σ_グループ [寄与(t) − 寄与(t−lag)]。ベースラインは差で消える。

    ts は contribution_timeseries の出力(t−lag の日も含む期間で計算しておく)。
    """
    rows = []
    for t in dates:
        s = t - pd.Timedelta(days=lag)
        if s not in ts.index:
            continue
        d = ts.loc[t, groups] - ts.loc[s, groups]
        rows.append({"日付": t, "比較日": s, "予測値(比較日)": ts.loc[s, "予測値"],
                     "予測値(当日)": ts.loc[t, "予測値"],
                     "Δ予測値": ts.loc[t, "予測値"] - ts.loc[s, "予測値"],
                     **{f"Δ{g}": v for g, v in d.items()}, "ΣΔ寄与": d.sum(),
                     "Δ実測値": ts.loc[t, "実測値"] - ts.loc[s, "実測値"]})
    out = pd.DataFrame(rows).set_index("日付")
    out["恒等式の誤差"] = (out["Δ予測値"] - out["ΣΔ寄与"]).abs()
    return out


def main_driver(row: pd.Series, groups: list[str]) -> str:
    """変化の方向に最も大きく寄与したグループ(主因)。"""
    d = row[[f"Δ{g}" for g in groups]]
    d = d if row["Δ予測値"] >= 0 else -d
    return d.idxmax()[1:]


@produces_table
def top_change_drivers(delta: pd.DataFrame, groups: list[str], n: int
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """上昇・下落の上位 n 日の一覧と、主因グループの集計(件数と平均Δ寄与)。"""
    up = delta.nlargest(n, "Δ予測値").assign(方向="上昇")
    down = delta.nsmallest(n, "Δ予測値").assign(方向="下落")
    top = pd.concat([up, down])
    top["主因グループ"] = [main_driver(r, groups) for _, r in top.iterrows()]
    top.insert(0, "曜日付き", [date_label(d) for d in top.index])
    counts = (top.groupby(["方向", "主因グループ"]).size().unstack(fill_value=0)
              .reindex(columns=groups, fill_value=0))
    means = top.groupby("方向")[[f"Δ{g}" for g in groups] + ["Δ予測値"]].mean()
    summary = pd.concat([counts.add_prefix("主因の件数:"), means.add_prefix("平均")], axis=1)
    return top, summary
