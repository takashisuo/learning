"""このデータ固有の What-if 設定: config のシナリオを読み、外生変数以外を動かすシナリオを拒否する。

再予測・ΔSHAP の計算は ShapAnalyzer.whatif / whatif_single / whatif_period が行う。
- 動かすのは外生変数のみ。自己回帰特徴量(ラグ・移動統計)は現実の履歴と矛盾するので書き換えない。
- 外生変数から派生した特徴量は現在の設計に無い(atemp_c は除外済み)。追加した場合は再計算が必要。
- level 変換の補正項(roll7_mean)は外生変数を動かしても変わらないので、Δ予測値 = Σ ΔSHAP が成り立つ。
"""
from __future__ import annotations

from .xai_platform import Scenario


def parse_scenarios(spec: dict, exog: list[str]) -> dict[str, Scenario]:
    """config の {名前: {特徴量: [種類, 量]}} を Scenario に変換し、外生変数以外を拒否する。"""
    out = {}
    for name, changes in spec.items():
        bad = [f for f in changes if f not in exog]
        if bad:
            raise ValueError(f"{name}: 外生変数以外 {bad} は動かさない(自己回帰特徴量の書き換えは禁止)")
        out[name] = {f: (k, float(a)) for f, (k, a) in changes.items()}
    return out
