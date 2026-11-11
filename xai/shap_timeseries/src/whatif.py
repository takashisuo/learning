"""外生変数の What-if: (A) モデル再予測による Δ予測値 と (B) ΔSHAP による内訳。

- 動かすのは外生変数のみ。自己回帰特徴量(ラグ・移動統計)は現実の履歴と矛盾するので書き換えない。
- 外生変数から派生した特徴量は現在の設計に無い(atemp_c は除外済み)。追加した場合は
  derived_features で再計算すること。
- level 変換では roll7_mean は外生変数を動かしても変わらないので、
  元スケールの Δ予測値 = モデルスケールの Δ = Σ ΔSHAP が成り立つ。
- 単一日のシナリオは xai_platform.whatif.scenario_analysis をそのまま使う。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .explain import OriginalScaleExplainer
from .xai_platform.provenance import produces_table
from .xai_platform.whatif import (Scenario, feature_bounds, resolve_delta, robust_sd,
                                  scenario_analysis)


def parse_scenarios(spec: dict, exog: list[str]) -> dict[str, Scenario]:
    """config の {名前: {特徴量: [種類, 量]}} を Scenario に変換し、外生変数以外を拒否する。"""
    out = {}
    for name, changes in spec.items():
        bad = [f for f in changes if f not in exog]
        if bad:
            raise ValueError(f"{name}: 外生変数以外 {bad} は動かさない(自己回帰特徴量の書き換えは禁止)")
        out[name] = {f: (k, float(a)) for f, (k, a) in changes.items()}
    return out


class WhatIf:
    def __init__(self, ex: OriginalScaleExplainer, X_train: pd.DataFrame, exog: list[str],
                 quantiles: tuple[float, float], correlated_pairs=()):
        self.ex, self.model = ex, ex.model
        lo, hi = quantiles
        self.sd = robust_sd(X_train, lo, hi)
        self.bounds = feature_bounds(X_train, lo, hi)
        self.exog = exog
        self.pairs = [tuple(p) for p in correlated_pairs]

    # ------------------------------------------------------------ 単一日
    def day(self, x: pd.Series, scenarios: dict[str, Scenario]):
        """1日分のシナリオ分析(summary / dshap / changes)。dshap はモデルスケール = 元スケールの ΔSHAP。"""
        return scenario_analysis(self.model, self.ex.tree, x, self.sd, self.bounds, scenarios,
                                 self.pairs)

    def tornado_frame(self, x: pd.Series, singles: dict[str, Scenario]) -> pd.DataFrame:
        """単独変更シナリオの結果を xai_platform.plots.tornado の入力形式にする。"""
        summary, _, changes = self.day(x, singles)
        df = changes.merge(summary[["シナリオ", "Δ予測値(同時)", "ΣΔSHAP", "相関注記"]],
                           on="シナリオ")
        df = df.rename(columns={"Δ予測値(同時)": "Δ予測値"})
        df["特徴量"] = df["シナリオ"]
        return df.reindex(df["Δ予測値"].abs().sort_values(ascending=False).index
                          ).reset_index(drop=True)

    # ------------------------------------------------------------ 期間全体
    def apply_frame(self, X: pd.DataFrame, changes: Scenario) -> tuple[pd.DataFrame, pd.Series]:
        """全日に変更を適用し、訓練期間の範囲にクリップする。返り値はクリップが起きた日のフラグ。"""
        new = X.copy()
        clipped = pd.Series(False, index=X.index)
        for f, (kind, amount) in changes.items():
            req = X[f] + np.array([resolve_delta(kind, amount, v, self.sd[f]) for v in X[f]])
            lo, hi = self.bounds.loc[f].iloc[0], self.bounds.loc[f].iloc[1]
            new[f] = req.clip(lo, hi)
            clipped |= ~np.isclose(req, new[f])
        return new, clipped

    @produces_table
    def period(self, X: pd.DataFrame, scenarios: dict[str, Scenario]
               ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """期間全体に適用したシナリオの総量の変化(summary / 日別Δ / 月別平均Δ / 特徴量別平均ΔSHAP)。"""
        p0 = self.model.predict(X)
        sv0 = self.ex.raw(X).values
        summary, daily, dshap = [], {}, {}
        for name, changes in scenarios.items():
            new, clipped = self.apply_frame(X, changes)
            d = self.model.predict(new) - p0
            ds = self.ex.raw(new).values - sv0
            assert np.allclose(d, ds.sum(axis=1), atol=1e-6), "Δ予測値 ≠ ΣΔSHAP"
            singles = sum(self.model.predict(self.apply_frame(X, {f: c})[0]) - p0
                          for f, c in changes.items())
            daily[name] = pd.Series(d, index=X.index)
            dshap[name] = pd.Series(ds.mean(axis=0), index=X.columns)
            summary.append({"シナリオ": name, "日数": len(X),
                            "元の予測の合計": p0.sum(), "Δ予測値の合計": d.sum(),
                            "Δ予測値の合計(%)": d.sum() / p0.sum() * 100,
                            "1日平均Δ": d.mean(), "1日Δの最小": d.min(), "1日Δの最大": d.max(),
                            "単独Δの合計の1日平均": np.mean(singles),
                            "相互作用分の1日平均": np.mean(d - singles),
                            "クリップした日数": int(clipped.sum())})
        daily = pd.DataFrame(daily)
        monthly = daily.groupby(daily.index.to_period("M")).mean()
        monthly.index = monthly.index.astype(str)
        monthly.index.name = "月"
        return pd.DataFrame(summary).set_index("シナリオ"), daily, monthly, pd.DataFrame(dshap)
