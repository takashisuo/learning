"""
営業員数を使った「信頼度（クレディビリティ）ウェイト」による支社別シュリンク予測。

考え方:
    支社の最終比率 = w × (支社自身の比率) + (1-w) × (統括部の比率)
    w = 営業員数 ÷ (営業員数 + k)

営業員数が多い支社ほど、自社の実績パターンを強く信頼する(wが1に近づく)。
営業員数が少ない支社ほど、統括部平均に強く引き戻される(wが0に近づく)。
kは「wがちょうど0.5になる営業員数」を表す調整用の定数で、データから学習するのではなく
業務判断で決める値（今回は支社の営業員数の中央値付近を仮置き）。

このスクリプトでは:
  1. 5月分の実績から、統括部平均の比率と、支社ごと自身の比率の両方を求める
  2. 6月分の営業員数からwを計算し、シュリンクした比率を作る
  3. 「統括部平均だけを使う予測（従来）」と「シュリンクした予測（今回）」を
     6月分の実際の月末実績と比較し、精度が改善するか検証する
"""

import numpy as np
import pandas as pd

from forecast_curve_example import add_day_offsets

K = 75  # wが0.5になる営業員数（支社の営業員数の中央値付近を仮置き）


def shrink_ratio(branch_ratio: float, dept_ratio: float, staff: float, k: float = K) -> float:
    """支社自身の比率と統括部平均の比率を、営業員数に応じてブレンドする"""
    w = staff / (staff + k)
    return w * branch_ratio + (1 - w) * dept_ratio


def main() -> None:
    df = pd.read_csv("sample_sales.csv")
    df = add_day_offsets(df)

    df_calib = df[df["月分"] == 5].set_index("機関番号")
    df_target = df[df["月分"] == 6].copy()

    # --- 統括部平均の比率（従来方式） ---
    dept_curve = df_calib.groupby("マーケット統括部名").apply(
        lambda d: pd.Series({
            "dept_spike_ratio": (d["21日時点成績"] / d["月末実績"]).mean(),
            "dept_day1_ratio": (d["1日時点成績"] / d["月末実績"]).mean(),
        })
    )

    def compute_row(row):
        dept = row["マーケット統括部名"]
        dept_spike = dept_curve.loc[dept, "dept_spike_ratio"]
        dept_day1 = dept_curve.loc[dept, "dept_day1_ratio"]

        # 支社自身の比率（5月分・自分の実績のみから）
        calib_row = df_calib.loc[row["機関番号"]]
        branch_spike = calib_row["21日時点成績"] / calib_row["月末実績"]
        branch_day1 = calib_row["1日時点成績"] / calib_row["月末実績"]

        staff = row["営業員数"]
        w = staff / (staff + K)
        shrunk_spike = shrink_ratio(branch_spike, dept_spike, staff)
        shrunk_day1 = shrink_ratio(branch_day1, dept_day1, staff)

        forecast_baseline = row["21日時点成績"] / dept_spike       # 従来：統括部平均のみ
        forecast_shrink = row["21日時点成績"] / shrunk_spike        # 今回：シュリンク後

        return pd.Series({
            "w": w,
            "予測_従来(統括部平均のみ)": forecast_baseline,
            "予測_シュリンク後": forecast_shrink,
        })

    df_target[["w", "予測_従来(統括部平均のみ)", "予測_シュリンク後"]] = df_target.apply(compute_row, axis=1)

    df_target["誤差率_従来(%)"] = (
        (df_target["予測_従来(統括部平均のみ)"] - df_target["月末実績"]) / df_target["月末実績"] * 100
    )
    df_target["誤差率_シュリンク後(%)"] = (
        (df_target["予測_シュリンク後"] - df_target["月末実績"]) / df_target["月末実績"] * 100
    )

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)

    print("=== 支社別の例（中部、営業員数の少ない順） ===")
    sample = df_target[df_target["マーケット統括部名"] == "中部"].sort_values("営業員数")
    print(sample[[
        "機関番号", "営業員数", "w", "月末実績",
        "予測_従来(統括部平均のみ)", "予測_シュリンク後",
        "誤差率_従来(%)", "誤差率_シュリンク後(%)",
    ]].round(2).to_string(index=False))

    print()
    print("=== 統括部ごとの平均絶対誤差率（MAPE） ===")
    df_target["絶対誤差率_従来"] = df_target["誤差率_従来(%)"].abs()
    df_target["絶対誤差率_シュリンク後"] = df_target["誤差率_シュリンク後(%)"].abs()
    mape = df_target.groupby("マーケット統括部名")[["絶対誤差率_従来", "絶対誤差率_シュリンク後"]].mean()
    print(mape.round(2))

    print()
    print("=== 全体のMAPE ===")
    print("従来(統括部平均のみ):", round(df_target["絶対誤差率_従来"].mean(), 2), "%")
    print("シュリンク後        :", round(df_target["絶対誤差率_シュリンク後"].mean(), 2), "%")


if __name__ == "__main__":
    main()
