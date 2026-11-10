"""
予測の信頼区間（誤差の幅）を、支社間のばらつきからブートストラップ法で算出する例。

時系列としては5月・6月の2ヶ月分しかなく、月をまたいだばらつきから統計的な
信頼区間を作るには心もとない。一方、1統括部あたり支社は約20あるため、
「支社ごとの比率のばらつき」を使えば、横方向のサンプル数を活かして
データ駆動の信頼区間を作れる。

やり方：
  1. 統括部内の支社を復元抽出（ブートストラップ）してリサンプル
  2. リサンプルした支社集合でスパイク比率・1日比率を再計算（＝カーブの再校正）
  3. そのカーブで着地予測を計算
  4. 1〜3をN回繰り返し、予測値の分布から信頼区間（パーセンタイル）を求める
"""

import numpy as np
import pandas as pd

from forecast_curve_example import add_day_offsets

RNG = np.random.default_rng(seed=123)
N_BOOTSTRAP = 2000


def bootstrap_forecast_interval(
    df_calib: pd.DataFrame,
    dept: str,
    cum_actual: float,
    day_offset: float,
    checkpoint: str,  # "spike" or "day1"
    n_boot: int = N_BOOTSTRAP,
) -> tuple[float, float, float]:
    """支社リサンプリングによる着地予測の点推定と信頼区間(10%/90%タイル)を返す"""
    d = df_calib[df_calib["マーケット統括部名"] == dept]

    if checkpoint == "spike":
        ratios = (d["21日時点成績"] / d["月末実績"]).to_numpy()
        offsets = np.zeros(len(d))
    else:
        ratios = (d["1日時点成績"] / d["月末実績"]).to_numpy()
        offsets = d["経過日数_1日"].to_numpy()

    final_offsets = d["経過日数_月末"].to_numpy()

    n = len(d)
    boot_forecasts = np.empty(n_boot)
    for i in range(n_boot):
        idx = RNG.integers(0, n, size=n)  # 支社を復元抽出
        # リサンプルした支社集合でアンカー点（比率）を再計算
        ratio_resampled = ratios[idx].mean()
        offset_resampled = offsets[idx].mean()
        final_offset_resampled = final_offsets[idx].mean()

        xp = [0, offset_resampled, final_offset_resampled] if checkpoint == "day1" else [0, final_offset_resampled]
        fp = [ratio_resampled, ratio_resampled, 1.0] if checkpoint == "day1" else [ratio_resampled, 1.0]
        # day_offsetにおける比率を補間
        ratio_at_offset = np.interp(day_offset, xp, fp)
        boot_forecasts[i] = cum_actual / ratio_at_offset

    point = np.median(boot_forecasts)
    lower, upper = np.percentile(boot_forecasts, [10, 90])
    return point, lower, upper


def main() -> None:
    df = pd.read_csv("sample_sales.csv")
    df = add_day_offsets(df)

    df_calib = df[df["月分"] == 5]
    df_target = df[df["月分"] == 6]

    rows = []
    for dept in df["マーケット統括部名"].unique():
        d_target = df_target[df_target["マーケット統括部名"] == dept]
        actual_final = d_target["月末実績"].sum()

        cum_21 = d_target["21日時点成績"].sum()
        cum_1 = d_target["1日時点成績"].sum()
        day1_offset = d_target["経過日数_1日"].mean()

        p21, lo21, hi21 = bootstrap_forecast_interval(df_calib, dept, cum_21, 0, "spike")
        p1, lo1, hi1 = bootstrap_forecast_interval(df_calib, dept, cum_1, day1_offset, "day1")

        rows.append({
            "統括部": dept,
            "実績(月末)": actual_final,
            "予測_21日時点": round(p21),
            "区間_21日時点": f"[{lo21:.0f}, {hi21:.0f}]",
            "実績は区間内か_21日": lo21 <= actual_final <= hi21,
            "予測_1日時点": round(p1),
            "区間_1日時点": f"[{lo1:.0f}, {hi1:.0f}]",
            "実績は区間内か_1日": lo1 <= actual_final <= hi1,
        })

    result = pd.DataFrame(rows).set_index("統括部")
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print(result)


if __name__ == "__main__":
    main()
