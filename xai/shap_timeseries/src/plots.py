"""このデータ固有の作図(系列の推移・予測の比較・STL 比較)。

SHAP の図(ウォーターフォール・積み上げ推移・依存プロット・What-if など)は ShapAnalyzer が描く。
日本語フォント設定・保存・来歴記録は xai_platform と共通。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .xai_platform import Unit, plt, produces_figure  # plt の import で日本語フォントが設定される
from .xai_platform import save_figure as _save


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
