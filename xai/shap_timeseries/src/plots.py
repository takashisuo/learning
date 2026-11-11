"""時系列用の作図。日本語フォント設定・保存・来歴記録は xai_platform.plots と共通。

SHAP の汎用図(waterfall / beeswarm / dependence / tornado など)は xai_platform.plots をそのまま使う。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .xai_platform.plots import _save, plt  # noqa: F401  plt の import で日本語フォントが設定される
from .xai_platform.provenance import produces_figure
from .xai_platform.units import Unit


@produces_figure()
def series_with_split(y: pd.Series, test_start: pd.Timestamp, path: Path, title: str,
                      unit: Unit = Unit(), ma: int = 28) -> Path:
    """目的変数の推移と訓練/テストの境界。"""
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(y.index, y, lw=0.7, color="#9aa5b1", label="日次の実績")
    ax.plot(y.index, y.rolling(ma, center=True).mean(), lw=2, color="#1f4e79",
            label=f"{ma}日移動平均")
    ax.axvspan(test_start, y.index[-1], color="#f4a261", alpha=0.15, label="テスト期間")
    ax.set_title(title)
    ax.set_ylabel(unit.label)
    ax.legend(loc="upper left", frameon=False)
    return _save(fig, path)


@produces_figure()
def test_predictions(y_test: pd.Series, preds: dict[str, pd.Series], path: Path, title: str,
                     unit: Unit = Unit()) -> Path:
    """テスト期間の実測値と各モデルの予測値。"""
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(y_test.index, y_test, color="black", lw=1.6, label="実測")
    for name, p in preds.items():
        ax.plot(p.index, p, lw=1, label=name)
    ax.set_title(title)
    ax.set_ylabel(unit.label)
    ax.legend(loc="lower left", frameon=False, ncol=len(preds) + 1)
    return _save(fig, path)


def _group_colors(groups, colors: dict[str, str] | None) -> dict:
    cmap = plt.get_cmap("tab10")
    return {g: (colors or {}).get(g, cmap(k % 10)) for k, g in enumerate(groups)}


@produces_figure()
def steps_waterfall(start_label: str, start: float, steps: pd.Series, end_label: str,
                    title: str, path: Path, unit: Unit = Unit(),
                    note: str | None = None) -> Path:
    """「始点 + 各ステップ = 終点」の横向きウォーターフォール(グループ単位の分解・Δ分解に使う)。"""
    labels = [start_label] + list(steps.index) + [end_label]
    end = start + steps.sum()
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(labels) + 1.6))
    y = np.arange(len(labels))[::-1]
    ax.barh(y[0], start, color="#bbbbbb")
    ax.text(start, y[0], f" {unit.fmt(start)}", va="center", fontsize=9)
    cum = start
    for k, (g, v) in enumerate(steps.items(), start=1):
        ax.barh(y[k], v, left=cum, color="#d6604d" if v >= 0 else "#4393c3")
        ax.text(max(cum, cum + v), y[k], f" {unit.fmt(v, True)}", va="center", fontsize=9)
        ax.plot([cum + v] * 2, [y[k] - 0.4, y[k] - 0.6], color="gray", lw=0.8)
        cum += v
    ax.barh(y[-1], end, color="#444444")
    ax.text(end, y[-1], f" {unit.fmt(end)}", va="center", fontsize=9, fontweight="bold")
    ax.set_yticks(y, labels)
    lo = min(0, start, end, *(start + steps.cumsum()))
    hi = max(start, end, *(start + steps.cumsum()))
    ax.set_xlim(lo, hi + (hi - lo) * 0.18)
    ax.set_xlabel(unit.label)
    ax.set_title(title, fontsize=11)
    if note:
        ax.text(0, -0.12, note, transform=ax.transAxes, fontsize=8, color="#555", va="top")
    return _save(fig, path)


@produces_figure()
def stacked_contributions(ts: pd.DataFrame, groups: list[str], test_start: pd.Timestamp,
                          path: Path, title: str, unit: Unit = Unit(),
                          colors: dict[str, str] | None = None) -> Path:
    """ベースラインから正の寄与を上に、負の寄与を下に積み上げた面グラフ + 実測・予測の線。"""
    cols = _group_colors(groups, colors)
    base = float(ts["ベースライン"].iloc[0])
    x = ts.index
    fig, ax = plt.subplots(figsize=(13, 5.5))
    pos = np.full(len(ts), base)
    neg = np.full(len(ts), base)
    for g in groups:
        v = ts[g].to_numpy()
        p, n = np.clip(v, 0, None), np.clip(v, None, 0)
        ax.fill_between(x, pos, pos + p, color=cols[g], alpha=0.75, lw=0, label=g, step="mid")
        ax.fill_between(x, neg, neg + n, color=cols[g], alpha=0.75, lw=0, step="mid")
        pos, neg = pos + p, neg + n
    ax.axhline(base, color="k", ls="--", lw=1, label=f"ベースライン {unit.fmt(base)}")
    ax.plot(x, ts["実測値"], color="k", lw=0.9, label="実測値")
    ax.plot(x, ts["予測値"], color="#d62728", lw=1.2, label="予測値 = ベースライン + Σ寄与")
    ax.axvline(test_start, color="gray", lw=1)
    ax.text(test_start, ax.get_ylim()[1], " テスト期間→", va="top", fontsize=9, color="gray")
    ax.set_ylabel(unit.label)
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), fontsize=8, ncol=7, frameon=False)
    return _save(fig, path)


@produces_figure()
def profile_heatmap(df: pd.DataFrame, path: Path, title: str, unit: Unit = Unit()) -> Path:
    """行 = 季節/月/曜日、列 = グループ の平均寄与のヒートマップ(赤 = 押し上げ、青 = 押し下げ)。"""
    m = df.to_numpy(float)
    lim = np.abs(m).max()
    fig, ax = plt.subplots(figsize=(1.6 * df.shape[1] + 2, 0.42 * df.shape[0] + 1.6))
    im = ax.imshow(m, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(df.shape[1]), df.columns)
    ax.set_yticks(range(df.shape[0]), df.index)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, unit.fmt(m[i, j], True), ha="center", va="center", fontsize=8,
                    color="white" if abs(m[i, j]) > lim * 0.6 else "black")
    fig.colorbar(im, ax=ax, label=f"平均寄与({unit.label})")
    ax.set_ylabel(df.index.name or "")
    ax.set_title(title)
    return _save(fig, path)


@produces_figure()
def stl_comparison(weekly: pd.DataFrame, annual: pd.DataFrame, path: Path,
                   unit: Unit = Unit()) -> Path:
    """STL の週周期・年周期成分と、モデルの寄与(平均からのずれ)の比較。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    styles = ["k-o", "-s", "-^", "-v"]
    for ax, df, xl, ttl in [(axes[0], weekly, "曜日", "週周期: STL成分 vs カレンダー寄与(週内訳)"),
                            (axes[1], annual, "月", "年周期: STL成分 vs モデルの寄与")]:
        for k, c in enumerate(df.columns):
            ax.plot(range(len(df)), df[c], styles[k], label=c, lw=2 if k == 0 else 1.3, ms=4)
        ax.set_xticks(range(len(df)), df.index)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xlabel(xl)
        ax.set_ylabel(f"平均からのずれ({unit.label})")
        ax.set_title(ttl, fontsize=11)
        ax.legend(fontsize=8)
    return _save(fig, path)


@produces_figure()
def period_whatif(daily: pd.DataFrame, monthly: pd.DataFrame, path: Path, title: str,
                  unit: Unit = Unit()) -> Path:
    """期間全体に適用したシナリオの日別Δ予測値と月別平均。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw={"width_ratios": [3, 1.3]})
    for c in daily.columns:
        axes[0].plot(daily.index, daily[c], lw=1, label=c)
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_ylabel(f"Δ予測値({unit.label})")
    axes[0].set_title("日別のΔ予測値(モデル再予測)", fontsize=11)
    axes[0].legend(fontsize=8)
    monthly.plot.bar(ax=axes[1], rot=0)
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_ylabel(f"1日平均Δ予測値({unit.label})")
    axes[1].set_title("月別の平均Δ予測値", fontsize=11)
    axes[1].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, path)
