"""
「最終達成率 = 第一報達成率 + a*残営業日数 + b*営業員数」という提案式を
実際にサンプルデータで回帰し、係数(特にa)がどれだけ不安定かを検証する。

達成率は「自社の月末実績に対する進捗率」として定義する
（目標/ノルマに対する達成率データが無いため、既存の比率と同じ定義で代用）。

  第一報達成率 = 21日時点成績 ÷ 月末実績   (＝ spike_ratio)
  最終達成率   = 月末実績 ÷ 月末実績 = 1.0  (定義上、常に1)

  → 残差 = 最終達成率 - 第一報達成率 = 1 - spike_ratio
    を a*残営業日数 + b*営業員数 で説明できるかを回帰する
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("sample_sales.csv")
df["サイクル開始日"] = pd.to_datetime(df["サイクル開始日"])
df["サイクル終了日"] = pd.to_datetime(df["サイクル終了日"])

df["第一報達成率"] = df["21日時点成績"] / df["月末実績"]
df["最終達成率"] = 1.0  # 定義上、月末実績÷月末実績は常に1
df["残差"] = df["最終達成率"] - df["第一報達成率"]
df["残営業日数"] = (df["サイクル終了日"] - df["サイクル開始日"]).dt.days

print("=== 残営業日数のバラつき（回帰の説明変数として使えるか） ===")
print(df.groupby("月分")["残営業日数"].agg(["mean", "std", "nunique"]))
print()

# ------------------------------------------------------------------
# ケース1: 5月分だけで回帰（＝残営業日数が単一の値しかない状態）
# ------------------------------------------------------------------
print("=" * 60)
print("ケース1: 5月分のみ(100支社)で回帰 → 残営業日数は全行30で完全に一定")
print("=" * 60)
d1 = df[df["月分"] == 5]
X1 = sm.add_constant(d1[["残営業日数", "営業員数"]])
try:
    model1 = sm.OLS(d1["残差"], X1).fit()
    print(model1.summary())
except Exception as e:
    print("回帰が実行できませんでした:", e)

# ------------------------------------------------------------------
# ケース2: 5月分+6月分をプールして回帰（残営業日数は30と29の2値のみ）
# ------------------------------------------------------------------
print()
print("=" * 60)
print("ケース2: 5月分+6月分(200支社)をプール → 残営業日数は30 or 29の2値のみ")
print("=" * 60)
X2 = sm.add_constant(df[["残営業日数", "営業員数"]])
model2 = sm.OLS(df["残差"], X2).fit()
print(model2.summary())

# ------------------------------------------------------------------
# ケース3: ブートストラップで係数aがどれだけブレるかを可視化
# ------------------------------------------------------------------
print()
print("=" * 60)
print("ケース3: 支社をリサンプリングして回帰を1000回繰り返し、係数aの分布を見る")
print("=" * 60)
RNG = np.random.default_rng(0)
n = len(df)
a_estimates = []
b_estimates = []
for _ in range(1000):
    idx = RNG.integers(0, n, size=n)
    d_boot = df.iloc[idx]
    X_boot = sm.add_constant(d_boot[["残営業日数", "営業員数"]])
    try:
        m = sm.OLS(d_boot["残差"].to_numpy(), X_boot.to_numpy()).fit()
        a_estimates.append(m.params[1])
        b_estimates.append(m.params[2])
    except Exception:
        continue

a_estimates = np.array(a_estimates)
b_estimates = np.array(b_estimates)
print(f"係数a(残営業日数)の分布: 中央値={np.median(a_estimates):.5f}, "
      f"標準偏差={np.std(a_estimates):.5f}, "
      f"10-90%タイル=[{np.percentile(a_estimates,10):.5f}, {np.percentile(a_estimates,90):.5f}]")
print(f"係数b(営業員数)の分布 : 中央値={np.median(b_estimates):.5f}, "
      f"標準偏差={np.std(b_estimates):.5f}, "
      f"10-90%タイル=[{np.percentile(b_estimates,10):.5f}, {np.percentile(b_estimates,90):.5f}]")
print(f"符号が反転した割合(a): {(np.sign(a_estimates) != np.sign(np.median(a_estimates))).mean()*100:.1f}%")
