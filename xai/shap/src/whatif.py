"""What-if シミュレーション: モデル再予測による変化量と、ΔSHAP による分解。

SHAP値は「現在の値と平均との差による寄与」であり「値を動かしたときの変化量」ではない。
変化量の根拠は常に model.predict の再実行(A)とし、SHAP はその説明(B)に使う。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .explain import compute_shap

LOWER_Q, UPPER_Q = 0.01, 0.99  # 動かしてよい範囲 = 訓練データの 1〜99 パーセンタイル

# 変更の指定: (種類, 量)。種類は "sd"(頑健σの倍数) / "abs"(元の単位で加算) / "pct"(割合)
# 複数変数シナリオ(計画で合意済み)
SCENARIOS = {
    "S1 所得と広さが上がる": {"MedInc": ("sd", 1.0), "AveRooms": ("sd", 1.0)},
    "S2 築浅・世帯人数が少ない": {"HouseAge": ("abs", -10.0), "AveOccup": ("abs", -0.5)},
    "S3 海岸沿いに北西へ移動": {"Latitude": ("abs", 1.0), "Longitude": ("abs", -1.0)},
    "S4 3変数を同時に変更": {"MedInc": ("sd", 1.0), "HouseAge": ("abs", -10.0),
                         "AveOccup": ("abs", -0.5)},
}

# +10% を適用しない特徴量(緯度経度の10%は約3.5度で意味をなさない)
PCT_EXCLUDED = ("Latitude", "Longitude")

# 片方だけ動かすと非現実的になりうる相関ペア
CORRELATED_PAIRS = [("Latitude", "Longitude"), ("AveRooms", "AveBedrms")]


def feature_bounds(X_train: pd.DataFrame) -> pd.DataFrame:
    return X_train.quantile([LOWER_Q, UPPER_Q]).T.set_axis(["下限(1%)", "上限(99%)"], axis=1)


def robust_sd(X_train: pd.DataFrame) -> pd.Series:
    """1〜99 パーセンタイルの範囲に収まるデータで計算した標準偏差(外れ値で膨らまない)。"""
    lo, hi = X_train.quantile(LOWER_Q), X_train.quantile(UPPER_Q)
    return X_train.apply(lambda c: c[(c >= lo[c.name]) & (c <= hi[c.name])].std())


def resolve_delta(kind: str, amount: float, current: float, sd: float) -> float:
    if kind == "sd":
        return amount * sd
    if kind == "abs":
        return amount
    if kind == "pct":
        return amount * current
    raise ValueError(kind)


def apply_changes(x: pd.Series, changes: dict, sd: pd.Series,
                  bounds: pd.DataFrame) -> tuple[pd.Series, list[dict]]:
    """変更を適用し、1〜99%範囲にクリップする。範囲外の要求は警告として返す。"""
    new = x.copy()
    log = []
    for f, (kind, amount) in changes.items():
        requested = x[f] + resolve_delta(kind, amount, x[f], sd[f])
        lo, hi = bounds.loc[f]
        applied = float(np.clip(requested, lo, hi))
        new[f] = applied
        log.append({"特徴量": f, "元の値": x[f], "要求値": requested, "適用値": applied,
                    "外挿警告": not np.isclose(requested, applied)})
    return new, log


def _predict_one(model, row: pd.Series) -> float:
    return float(model.predict(row.to_frame().T.astype(float))[0])


def single_variable_whatif(model, x: pd.Series, sd: pd.Series, bounds: pd.DataFrame,
                           kind: str = "sd", amount: float = 1.0) -> pd.DataFrame:
    """各特徴量を単独で動かしたときの Δ予測値(トルネード図用)。"""
    base_pred = _predict_one(model, x)
    rows = []
    for f in x.index:
        if kind == "pct" and f in PCT_EXCLUDED:
            continue
        new, log = apply_changes(x, {f: (kind, amount)}, sd, bounds)
        new_pred = _predict_one(model, new)
        partners = [b if a == f else a for a, b in CORRELATED_PAIRS if f in (a, b)]
        rows.append({**log[0], "変更量": log[0]["適用値"] - log[0]["元の値"],
                     "元の予測": base_pred, "変更後の予測": new_pred,
                     "Δ予測値": new_pred - base_pred,
                     "相関注記": f"{partners[0]}と強相関(片方のみ変更)" if partners else ""})
    df = pd.DataFrame(rows)
    return df.reindex(df["Δ予測値"].abs().sort_values(ascending=False).index,
                      ).reset_index(drop=True)


def scenario_analysis(model, explainer, x: pd.Series, sd: pd.Series, bounds: pd.DataFrame,
                      scenarios: dict = SCENARIOS) -> tuple[pd.DataFrame, pd.DataFrame,
                                                          pd.DataFrame]:
    """複数変数シナリオ。

    返り値:
      summary: シナリオ別の Δ予測値・単独Δの合計・相互作用分(非加法分)・ΣΔSHAP
      dshap:   シナリオ×特徴量 の ΔSHAP(行和 = Δ予測値)
      changes: 適用した変更と外挿警告
    """
    base_pred = _predict_one(model, x)
    sv_before = compute_shap(explainer, x.to_frame().T.astype(float)).values[0]
    summary, dshap_rows, change_rows = [], {}, []
    for name, changes in scenarios.items():
        new, log = apply_changes(x, changes, sd, bounds)
        joint = _predict_one(model, new) - base_pred
        # 単独で動かした Δ の合計(同じクリップ後の値を使う)
        singles = {}
        for entry in log:
            one = x.copy()
            one[entry["特徴量"]] = entry["適用値"]
            singles[entry["特徴量"]] = _predict_one(model, one) - base_pred
        sv_after = compute_shap(explainer, new.to_frame().T.astype(float)).values[0]
        d = sv_after - sv_before
        dshap_rows[name] = d
        corr_note = [f"{a}/{b}" for a, b in CORRELATED_PAIRS
                     if (a in changes) != (b in changes)]
        summary.append({"シナリオ": name, "元の予測": base_pred, "変更後の予測": base_pred + joint,
                        "Δ予測値(同時)": joint, "単独Δの合計": sum(singles.values()),
                        "相互作用分(同時−単独合計)": joint - sum(singles.values()),
                        "ΣΔSHAP": d.sum(),
                        "単独Δの内訳": ", ".join(f"{k}:{v:+.3f}" for k, v in singles.items()),
                        "外挿警告": any(e["外挿警告"] for e in log),
                        "相関注記": "片方のみ変更: " + ", ".join(corr_note) if corr_note else ""})
        change_rows += [{"シナリオ": name, **e} for e in log]
    dshap = pd.DataFrame(dshap_rows, index=x.index).T
    return pd.DataFrame(summary), dshap, pd.DataFrame(change_rows)
