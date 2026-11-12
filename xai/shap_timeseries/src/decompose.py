"""このデータ固有の時間方向の処理: 季節/月/曜日のキー・STL 比較・比較時点(前日/前週/前年)ごとの差の分解。

グループ別寄与の集約・2時点の差の分解そのものは ShapAnalyzer(xai_platform)が行う。
"""
from __future__ import annotations

import pandas as pd
from statsmodels.tsa.seasonal import MSTL

from .explain import WEEKDAY_JA, date_label
from .xai_platform import ShapAnalyzer, produces_table

SEASON_JA = {12: "冬", 1: "冬", 2: "冬", 3: "春", 4: "春", 5: "春",
             6: "夏", 7: "夏", 8: "夏", 9: "秋", 10: "秋", 11: "秋"}


# ---------------------------------------------------------------- 区分のキー
def calendar_keys(index: pd.DatetimeIndex, by: str) -> tuple[object, str]:
    """ShapAnalyzer.profile に渡す (行ごとのキー, キー名)。by は season / month / weekday。"""
    if by == "season":
        return pd.Categorical(index.month.map(SEASON_JA), ["春", "夏", "秋", "冬"]), "季節"
    if by == "month":
        return pd.Index(index.month, name="月"), "月"
    if by == "weekday":
        return pd.Categorical([WEEKDAY_JA[(d + 1) % 7] for d in index.dayofweek], WEEKDAY_JA), "曜日"
    raise ValueError(by)


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
    wk_key, _ = calendar_keys(idx, "weekday")
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


# ---------------------------------------------------------------- 比較時点ごとの差の分解
def delta_decomposition(az: ShapAnalyzer, lag: int, dates: pd.Index) -> pd.DataFrame:
    """各日 t と t−lag 日の差の分解(ShapAnalyzer.compare_table を日付で呼ぶ)。"""
    pairs = [(t, t - pd.Timedelta(days=lag)) for t in dates]
    out = az.compare_table(pairs)
    out.index.name = "日付"
    return out.rename(columns={"比較対象": "比較日", "予測値(比較対象)": "予測値(比較日)",
                               "予測値(対象)": "予測値(当日)"})


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
