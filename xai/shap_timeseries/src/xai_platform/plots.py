"""作図(日本語フォント設定含む)。

すべての関数は保存先 path を受け取り、PNG(dpi=150)で保存してその path を返す。
単位(Unit)・タイトル・グループの色などデータ固有の情報は引数で渡す。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import japanize_matplotlib  # noqa: E402,F401  日本語フォント設定
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from .provenance import produces_figure  # noqa: E402
from .units import Unit  # noqa: E402

DPI = 150
plt.rcParams["axes.unicode_minus"] = False


def _save(fig, path: Path) -> Path:
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _japanize_shap_axes(fig, xlabel: str | None = None, colorbar: str | None = None) -> None:
    """shap が付ける英語ラベルを日本語に差し替える。"""
    fig.canvas.draw()  # 目盛ラベルの文字列を確定させる
    for ax in fig.axes:
        if ax.get_label() == "<colorbar>":
            if colorbar:
                ax.set_ylabel(colorbar)
            ticks = ax.get_yticks()
            labels = [{"High": "高", "Low": "低"}.get(t.get_text(), t.get_text())
                      for t in ax.get_yticklabels()]
            ax.set_yticks(ticks, labels)
    if xlabel:
        fig.axes[0].set_xlabel(xlabel)


def _colors(names, colors: dict[str, str] | None):
    cmap = plt.get_cmap("tab10")
    return {n: (colors or {}).get(n, cmap(k % 10)) for k, n in enumerate(names)}


# ---------------------------------------------------------------- ミクロ(shap.plots を使用)
@produces_figure(shap_api="shap.plots.waterfall")
def waterfall(sv_row: shap.Explanation, title: str, path: Path, unit: Unit = Unit(),
              max_display: int = 10) -> Path:
    """1サンプルの予測値を「ベースライン + 各特徴量の寄与」に分解したウォーターフォール図。"""
    shap.plots.waterfall(sv_row, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.axes[0].set_title(title + "\n下端 E[f(X)] = ベースライン(訓練データの平均予測)、"
                          "上端 f(x) = 予測値", fontsize=11, pad=28)
    fig.axes[0].set_xlabel(f"予測値への寄与の積み上げ({unit.axis})", labelpad=22)
    return _save(fig, path)


# ---------------------------------------------------------------- What-if
@produces_figure()
def tornado(df: pd.DataFrame, title: str, path: Path, unit: Unit = Unit()) -> Path:
    """各特徴量を単独で動かしたときの Δ予測値(再予測)のトルネード図。入力は whatif.single_variable_whatif。"""
    d = df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(d) + 1.5))
    colors = ["#d6604d" if v >= 0 else "#4393c3" for v in d["Δ予測値"]]
    ax.barh(d["特徴量"], d["Δ予測値"], color=colors)
    for i, (_, r) in enumerate(d.iterrows()):
        mark = " ※範囲外→クリップ" if r["外挿警告"] else ""
        label = f"{unit.fmt(r['Δ予測値'], True)}  ({r['元の値']:.4g}→{r['適用値']:.4g}){mark}"
        ax.text(r["Δ予測値"], i, " " + label if r["Δ予測値"] >= 0 else label + " ",
                va="center", ha="left" if r["Δ予測値"] >= 0 else "right", fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    lim = max(abs(d["Δ予測値"]).max() * 2.2, 0.05)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel(f"Δ予測値(モデル再予測、{unit.axis})")
    ax.set_title(title)
    return _save(fig, path)


@produces_figure()
def scenario_dshap(summary: pd.DataFrame, dshap: pd.DataFrame, title: str, path: Path,
                   unit: Unit = Unit()) -> Path:
    """複数変数シナリオの Δ予測値 = ΣΔSHAP の内訳と相互作用分の図。入力は whatif.scenario_analysis。

    左: Δ予測値 = ΣΔSHAP の特徴量別積み上げ、右: 同時変更Δ と 単独Δの合計 の比較。
    シナリオ名は「ID 説明」(空白区切り)を想定し、ID を軸ラベルの1行目に使う。
    """
    feats = dshap.columns
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [3, 2]})
    ax = axes[0]
    x = np.arange(len(dshap))
    pos_bottom = np.zeros(len(dshap))
    neg_bottom = np.zeros(len(dshap))
    for k, f in enumerate(feats):
        v = dshap[f].values
        b = np.where(v >= 0, pos_bottom, neg_bottom)
        ax.bar(x, v, bottom=b, color=cmap(k % 10), label=f, width=0.6)
        pos_bottom += np.where(v >= 0, v, 0)
        neg_bottom += np.where(v < 0, v, 0)
    ax.scatter(x, summary["Δ予測値(同時)"], color="k", marker="D", zorder=5,
               label="Δ予測値(再予測)")
    for i, v in enumerate(summary["Δ予測値(同時)"]):
        ax.annotate(unit.fmt(v, True), (i, v), textcoords="offset points", xytext=(12, 0),
                    fontsize=9, va="center")
    ax.axhline(0, color="k", lw=0.8)
    ids = [s.split(" ", 1) for s in dshap.index]
    ax.set_xticks(x, ["\n".join(p) for p in ids], fontsize=9)
    ax.set_ylabel(f"ΔSHAP({unit.axis})")
    ax.set_title("Δ予測値 = Σ ΔSHAP の特徴量別内訳")
    ax.legend(fontsize=8, ncol=2, loc="best")

    ax = axes[1]
    w = 0.38
    ax.bar(x - w / 2, summary["Δ予測値(同時)"], w, label="同時に変更したΔ", color="#444")
    ax.bar(x + w / 2, summary["単独Δの合計"], w, label="単独Δの合計", color="#aaa")
    for i, r in summary.iterrows():
        hi = max(r["Δ予測値(同時)"], r["単独Δの合計"], 0)
        lo = min(r["Δ予測値(同時)"], r["単独Δの合計"], 0)
        below = hi == 0 and lo < 0  # 棒が下向きなら棒の下に書く(タイトルとの重なりを避ける)
        ax.annotate(f"差 {unit.fmt(r['相互作用分(同時−単独合計)'], True)}", (i, lo if below else hi),
                    textcoords="offset points", xytext=(0, -4 if below else 4), ha="center",
                    va="top" if below else "bottom", fontsize=9)
    ax.margins(y=0.1)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, [p[0] for p in ids])
    ax.set_ylabel(f"Δ予測値({unit.axis})")
    ax.set_title("相互作用分 = 同時変更Δ − 単独Δの合計")
    ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, path)


# ---------------------------------------------------------------- マクロ
@produces_figure(shap_api="shap.plots.beeswarm")
def beeswarm(sv: shap.Explanation, path: Path, unit: Unit = Unit(),
             title: str = "各特徴量が予測値を押し上げる/押し下げる分布",
             max_display: int = 10) -> Path:
    """全サンプルのSHAP値の分布(色 = 特徴量の値の高低)。"""
    shap.plots.beeswarm(sv, max_display=max_display, show=False)
    fig = plt.gcf()
    _japanize_shap_axes(fig, xlabel=f"SHAP値(予測への寄与、{unit.axis})", colorbar="特徴量の値")
    fig.axes[0].set_title(title)
    return _save(fig, path)


@produces_figure()
def mean_abs_bar(sv: shap.Explanation, path: Path, unit: Unit = Unit(),
                 title: str = "特徴量ごとの寄与の平均的な大きさ") -> Path:
    """平均|SHAP|の棒グラフ(換算単位を併記)。"""
    imp = pd.Series(np.abs(sv.values).mean(axis=0), index=sv.feature_names).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(imp.index, imp.values, color="#2a78b5")
    for i, v in enumerate(imp.values):
        alt = f"(約{unit.to_alt(v):.1f}{unit.alt_label})" if unit.alt_label else ""
        ax.text(v, i, f" {unit.fmt(v)}{alt}", va="center", fontsize=8)
    ax.set_xlim(0, imp.max() * 1.45)
    ax.set_xlabel(f"平均|SHAP値|({unit.axis})")
    ax.set_title(title)
    return _save(fig, path)


@produces_figure(shap_api="shap.plots.scatter")
def dependence(sv: shap.Explanation, feature: str, color_feature: str, path: Path,
               unit: Unit = Unit(), xlim: tuple[float, float] | None = None) -> Path:
    """依存プロット(横軸 = 特徴量の値、縦軸 = SHAP値、色 = 相互作用の強い特徴量)。"""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    shap.plots.scatter(sv[:, feature], color=sv[:, color_feature], ax=ax, show=False,
                       alpha=0.5, dot_size=6)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel(f"{feature} の値")
    ax.set_ylabel(f"{feature} のSHAP値({unit.axis})")
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_title(f"{feature} の依存プロット(色: 相互作用の強い {color_feature})")
    for a in fig.axes[1:]:
        if a.get_label() == "<colorbar>":  # ヒストグラム用の副軸には付けない
            a.set_ylabel(f"{color_feature} の値")
    return _save(fig, path)


@produces_figure()
def interaction_heatmap(strength: pd.DataFrame, features: list[str], path: Path,
                        unit: Unit = Unit(),
                        title: str = "特徴量ペアの相互作用の強さ") -> Path:
    """特徴量ペアの相互作用の強さのヒートマップ。入力は explain.interaction_strength。"""
    m = pd.DataFrame(0.0, index=features, columns=features)
    for _, r in strength.iterrows():
        m.loc[r["特徴量1"], r["特徴量2"]] = m.loc[r["特徴量2"], r["特徴量1"]] = r["平均|相互作用|"]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(m.values, cmap="Oranges")
    ax.set_xticks(range(len(features)), features, rotation=45, ha="right")
    ax.set_yticks(range(len(features)), features)
    for i in range(len(features)):
        for j in range(len(features)):
            if i != j:
                ax.text(j, i, unit.fmt(m.values[i, j]), ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label=f"平均|相互作用SHAP値|({unit.axis})")
    ax.set_title(title)
    return _save(fig, path)


@produces_figure()
def interaction_pair(inter: np.ndarray, X: pd.DataFrame, f1: str, f2: str, path: Path,
                     unit: Unit = Unit()) -> Path:
    """2特徴量の相互作用SHAP値(主効果を除いた上乗せ分)の散布図。"""
    cols = list(X.columns)
    i, j = cols.index(f1), cols.index(f2)
    y = inter[:, i, j] + inter[:, j, i]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sc = ax.scatter(X[f1], y, c=X[f2], cmap="coolwarm", s=8, alpha=0.7)
    fig.colorbar(sc, ax=ax, label=f"{f2} の値")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel(f"{f1} の値")
    ax.set_ylabel(f"{f1}×{f2} の相互作用SHAP値({unit.axis})")
    ax.set_title(f"{f1} と {f2} の相互作用(主効果を除いた上乗せ分)")
    return _save(fig, path)


def _stack(ax, x, df: pd.DataFrame, base: float, width, colors: dict) -> None:
    pos = np.full(len(df), base)
    neg = np.full(len(df), base)
    for g in df.columns:
        v = df[g].values
        b = np.where(v >= 0, pos, neg)
        ax.bar(x, v, bottom=b, width=width, color=colors[g], label=g, linewidth=0)
        pos += np.where(v >= 0, v, 0)
        neg += np.where(v < 0, v, 0)


@produces_figure()
def group_stack_bins(bins_df: pd.DataFrame, groups: list[str], base: float, path: Path,
                     unit: Unit = Unit(), colors: dict[str, str] | None = None,
                     title: str = "予測値の構成: ベースライン + 特徴量グループ別の平均寄与") -> Path:
    """予測値ビンごとの「ベースライン + グループ別平均寄与」の積み上げ棒。入力は explain.group_by_prediction_bins。"""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(bins_df))
    _stack(ax, x, bins_df[groups], base, 0.7, _colors(groups, colors))
    ax.axhline(base, color="k", lw=1, ls="--", label=f"ベースライン {unit.fmt(base)}")
    ax.plot(x, bins_df["予測値"], "kD-", label="平均予測値")
    ax.set_xticks(x, [f"{i + 1}\n{r}" for i, r in enumerate(bins_df["予測値の範囲"])], fontsize=8)
    ax.set_xlabel(f"予測値の{len(bins_df)}分位ビン(低い→高い)と予測値の範囲"
                  if len(bins_df) != 10 else "予測値の十分位ビン(低い→高い)と予測値の範囲")
    ax.set_ylabel(f"予測値({unit.axis})")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=4, loc="upper left")
    return _save(fig, path)


@produces_figure()
def group_stack_sorted(group_df: pd.DataFrame, pred: np.ndarray, base: float, path: Path,
                       unit: Unit = Unit(), colors: dict[str, str] | None = None,
                       title: str = "全サンプルの予測値の構成(特徴量グループ別の寄与)",
                       xlabel: str = "サンプル(予測値の低い順に並べたもの)") -> Path:
    """全サンプルを予測値順に並べた、グループ別寄与の積み上げ図。"""
    order = np.argsort(pred)
    df = group_df.iloc[order].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    _stack(ax, np.arange(len(df)), df, base, 1.0, _colors(df.columns, colors))
    ax.plot(np.arange(len(df)), pred[order], color="k", lw=1, label="予測値")
    ax.axhline(base, color="k", lw=1, ls="--", label=f"ベースライン {unit.fmt(base)}")
    ax.set_xlim(0, len(df))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"予測値({unit.axis})")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=4, loc="upper left")
    return _save(fig, path)
