"""
CLAUDE.md の業務仕様に基づくサンプルCSV生成スクリプト。

想定データ構造:
- 機関番号: 支社を表す7桁の数値（約100支社）
- マーケット統括部名: 西日本/東日本/中部/首都圏/関西 のいずれか（支社は1つの統括部に固定所属）
- 月分: 5, 6（実績が存在する2ヶ月分）
- 21日時点成績: 当月21日時点の累積実績（15〜20日頃の先行計上分＝スパイクを含む）
- 1日時点成績: 翌月1日時点の累積実績（21日時点実績 + 22日〜月末+1日までの通常積み上げ分。21日時点より大きい値になる）
- 月末実績: サイクル終了時点（翌月20日前後）の最終着地実績。スパイク比率・累積比率カーブを算出するための正解値。
- サイクル開始日/1日時点日付/サイクル終了日: 各チェックポイントの実際の日付。
  t月21日〜t+1月20日という区切りは目安であり、月によって開始日が20日・21日等にブレる
  （5月分は5/20〜6/19だった、という業務側の補足に対応）。そのため経過日数は
  固定値ではなく、これらの実日付から都度計算する。
- 営業員数: その支社に所属する営業員の人数（業務側の補足：支社あたりおおよそ50〜100人、
  月をまたいだ変動は退職・異動により1〜2ヶ月に1人程度）。支社の比率をどれだけ信頼するか
  （シュリンクの重みw）を決めるための材料として使う。
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(seed=42)

DEPARTMENTS = ["西日本", "東日本", "中部", "首都圏", "関西"]
N_BRANCHES = 100
MONTHS = [5, 6]
YEAR = 2026

# 支社ごとの営業員数（業務側の補足：50〜100人、月をまたいだ変動は1〜2ヶ月に1人程度）
STAFF_MIN = 50
STAFF_MAX = 100

# サイクル開始日（21日固定ではなく、月によってブレることを再現）
# 5月分は業務側の補足通り5/20開始、6月分は基本形の6/21開始とする
CYCLE_START_DAY = {5: 20, 6: 21}


def cycle_dates(month: int) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """月分(month)から (サイクル開始日, 1日時点日付, サイクル終了日) を求める"""
    start = pd.Timestamp(year=YEAR, month=month, day=CYCLE_START_DAY[month])
    day1 = (start + pd.DateOffset(months=1)).replace(day=1)
    end = start + pd.DateOffset(months=1) - pd.Timedelta(days=1)
    return start, day1, end

# 統括部ごとの規模感（支社あたり月末着地の平均レンジ）と、スパイク比率のブレを変える
DEPT_PROFILE = {
    "西日本": {"base_mean": 320, "base_sd": 60, "spike_ratio_mean": 0.62},
    "東日本": {"base_mean": 410, "base_sd": 80, "spike_ratio_mean": 0.58},
    "中部":   {"base_mean": 250, "base_sd": 50, "spike_ratio_mean": 0.65},
    "首都圏": {"base_mean": 480, "base_sd": 90, "spike_ratio_mean": 0.55},
    "関西":   {"base_mean": 300, "base_sd": 55, "spike_ratio_mean": 0.60},
}


def make_institution_numbers(n: int) -> list[int]:
    """7桁の機関番号を重複なく生成"""
    numbers = RNG.choice(np.arange(1000000, 10000000), size=n, replace=False)
    return sorted(numbers.tolist())


def main() -> None:
    inst_numbers = make_institution_numbers(N_BRANCHES)

    # 支社を統括部へ均等割り当て（端数は先頭の統括部から+1ずつ）
    branch_depts = []
    base = N_BRANCHES // len(DEPARTMENTS)
    remainder = N_BRANCHES % len(DEPARTMENTS)
    for i, dept in enumerate(DEPARTMENTS):
        count = base + (1 if i < remainder else 0)
        branch_depts.extend([dept] * count)
    RNG.shuffle(branch_depts)

    rows = []
    for inst_no, dept in zip(inst_numbers, branch_depts):
        profile = DEPT_PROFILE[dept]
        # 支社ごとの月末着地水準（月をまたいでもある程度相関するよう支社固有のブレを先に決める）
        branch_level = max(50, RNG.normal(profile["base_mean"], profile["base_sd"]))

        # 支社ごとの営業員数の基準値（50〜100人）。月をまたぐ変動は退職・異動で±1〜2人程度
        staff = int(RNG.integers(STAFF_MIN, STAFF_MAX + 1))

        for month in MONTHS:
            if month != MONTHS[0]:
                staff = max(1, staff + int(RNG.integers(-2, 3)))
            start_date, day1_date, end_date = cycle_dates(month)

            # 月ごとの変動（前月比±10%程度）
            month_total = max(30, branch_level * RNG.normal(1.0, 0.08))

            # スパイク比率（統括部平均を中心に支社・月でブレを持たせる）
            spike_ratio = np.clip(
                RNG.normal(profile["spike_ratio_mean"], 0.05), 0.35, 0.85
            )

            day21_result = round(month_total * spike_ratio)

            # 22日〜翌月1日（約11日分）の通常積み上げ分
            normal_accumulation = round(max(0, month_total - day21_result) * RNG.uniform(0.35, 0.55))
            day1_result = day21_result + normal_accumulation

            # 月末実績は month_total を中心に少しノイズを乗せる（day1以降も積み上げが続くため）
            final_result = max(day1_result, round(month_total * RNG.normal(1.0, 0.03)))

            rows.append(
                {
                    "機関番号": inst_no,
                    "マーケット統括部名": dept,
                    "月分": month,
                    "21日時点成績": int(day21_result),
                    "1日時点成績": int(day1_result),
                    "月末実績": int(final_result),
                    "サイクル開始日": start_date.date().isoformat(),
                    "1日時点日付": day1_date.date().isoformat(),
                    "サイクル終了日": end_date.date().isoformat(),
                    "営業員数": staff,
                }
            )

    df = pd.DataFrame(rows)
    df = df.sort_values(["マーケット統括部名", "機関番号", "月分"]).reset_index(drop=True)

    out_path = "sample_sales.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"generated: {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
