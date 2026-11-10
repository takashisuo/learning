"""
CLAUDE.md の「累積比率カーブによる割り戻し方式」の実装例。

考え方:
    予測値 = 現時点までの累積実績 ÷ その経過日数における「対月末累積比率」

サイクルは t月21日 〜 t+1月20日 が目安だが、実際の開始日は月によって20日・21日等に
ブレる（5月分は5/20〜6/19だった、という業務側の補足に対応）。そのため「経過日数」は
0/11/30のような固定値ではなく、各サイクルの実際の日付（サイクル開始日・1日時点日付・
サイクル終了日）から都度計算する。

この経過日数(day_offset)に対して「その日までの累積実績が、最終的な月末着地の何%に
あたるか」という比率を求めたものが「累積比率カーブ」。カーブそのものは経過日数の
関数として作るので、開始日がズレても仕組みは変わらない。

本サンプルでは支社ごとの実績が薄い（2ヶ月分）ため、CLAUDE.md の方針通り
マーケット統括部（5部）単位でカーブを作る。

このスクリプトでは:
  1. 5月分の実績（21日時点・1日時点・月末実績）から、統括部ごとのカーブを校正する
  2. そのカーブを使って「6月分」の着地を、21日時点・1日時点それぞれのタイミングで予測する
     （ローリング再予測のシミュレーション）
  3. 6月分の実際の月末実績と比較して精度を確認する

※ 実運用では校正に使う月と予測対象の月が同じにならないようにする（本例は
   データが2ヶ月しかないため、5月で校正→6月を予測、という時系列順の1パターンのみ）。
"""

import numpy as np
import pandas as pd

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)


def add_day_offsets(df: pd.DataFrame) -> pd.DataFrame:
    """実日付（サイクル開始日・1日時点日付・サイクル終了日）から経過日数を計算する"""
    df = df.copy()
    start = pd.to_datetime(df["サイクル開始日"])
    df["経過日数_1日"] = (pd.to_datetime(df["1日時点日付"]) - start).dt.days
    df["経過日数_月末"] = (pd.to_datetime(df["サイクル終了日"]) - start).dt.days
    return df


def build_ratio_curve(df_calib: pd.DataFrame) -> pd.DataFrame:
    """校正用データ（実績確定済みの月）から統括部ごとの累積比率カーブのアンカー点を作る

    アンカー点は (経過日数, 比率) のペア。経過日数側も固定値ではなく実日付から
    計算した値の平均を使うため、開始日が月によってブレても正しく校正できる。
    """
    g = df_calib.groupby("マーケット統括部名")
    curve = pd.DataFrame(
        {
            "spike_ratio": g.apply(lambda d: (d["21日時点成績"] / d["月末実績"]).mean()),
            "day1_ratio": g.apply(lambda d: (d["1日時点成績"] / d["月末実績"]).mean()),
            "day1_offset": g["経過日数_1日"].mean(),
            "final_offset": g["経過日数_月末"].mean(),
        }
    )
    curve["spike_offset"] = 0  # 定義上、サイクル開始日=経過日数0
    curve["final_ratio"] = 1.0  # 定義上、月末時点の比率は必ず1.0
    return curve


def ratio_at(curve_row: pd.Series, day_offset: float) -> float:
    """経過日数(day_offset)における累積比率を、アンカー点間の線形補間で求める"""
    xp = [curve_row["spike_offset"], curve_row["day1_offset"], curve_row["final_offset"]]
    fp = [curve_row["spike_ratio"], curve_row["day1_ratio"], curve_row["final_ratio"]]
    return float(np.interp(day_offset, xp, fp))


def forecast_landing(cum_actual: float, day_offset: float, curve_row: pd.Series) -> float:
    """現時点までの累積実績を、その日の対月末累積比率で割り戻して月末着地を予測

    day_offset は予測対象サイクル自身の実日付から計算した値を渡す
    （校正元の月とはサイクル開始日がズレていてもよい）。
    """
    ratio = ratio_at(curve_row, day_offset)
    return cum_actual / ratio


def main() -> None:
    df = pd.read_csv("sample_sales.csv")
    df = add_day_offsets(df)

    # --- 1. 5月分の実績で統括部ごとの累積比率カーブを校正 ---
    # (5月分は5/20開始、経過日数_1日=12、経過日数_月末=30)
    df_calib = df[df["月分"] == 5]
    curve = build_ratio_curve(df_calib)
    print("=== 統括部ごとの累積比率カーブ（5月分の実績から校正） ===")
    print(curve.round(3))
    print()

    # --- 2. 6月分を「21日時点」「1日時点」それぞれで着地予測（ローリング） ---
    # (6月分は6/21開始、経過日数_1日=10 と、5月分とはズレている点に注意。
    #  校正した比率カーブに対して、6月分自身の実際の経過日数で補間する)
    df_target = df[df["月分"] == 6].copy()

    def predict_row(row, offset, col):
        curve_row = curve.loc[row["マーケット統括部名"]]
        return forecast_landing(row[col], offset, curve_row)

    df_target["予測_21日時点"] = df_target.apply(
        lambda r: predict_row(r, 0, "21日時点成績"), axis=1
    )
    df_target["予測_1日時点"] = df_target.apply(
        lambda r: predict_row(r, r["経過日数_1日"], "1日時点成績"), axis=1
    )

    # --- 3. 統括部単位に集計して、実際の月末実績と比較 ---
    summary = df_target.groupby("マーケット統括部名").agg(
        実績_月末=("月末実績", "sum"),
        予測_21日時点=("予測_21日時点", "sum"),
        予測_1日時点=("予測_1日時点", "sum"),
    )
    summary["誤差率_21日時点(%)"] = (
        (summary["予測_21日時点"] - summary["実績_月末"]) / summary["実績_月末"] * 100
    )
    summary["誤差率_1日時点(%)"] = (
        (summary["予測_1日時点"] - summary["実績_月末"]) / summary["実績_月末"] * 100
    )

    print("=== 6月分の着地予測 vs 実績（統括部集計） ===")
    print(summary.round(1))


if __name__ == "__main__":
    main()
