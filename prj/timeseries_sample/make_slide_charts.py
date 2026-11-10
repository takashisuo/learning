"""
スライド埋め込み用の図解(PNG)を3枚作る。
色は dataviz スキルの参照パレット(references/palette.md)に準拠。
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "Meiryo"
plt.rcParams["axes.unicode_minus"] = False

# --- パレット（dataviz skill の reference palette より） ---
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"
SERIES_BLUE = "#2a78d6"   # slot1
SERIES_AQUA = "#1baf7a"   # slot2
SERIES_YELLOW = "#eda100" # slot3
ORDINAL_LIGHT = "#86b6ef"  # step250
ORDINAL_MID = "#2a78d6"    # step450
ORDINAL_DARK = "#104281"   # step650

plt.rcParams["figure.facecolor"] = SURFACE
plt.rcParams["axes.facecolor"] = SURFACE
plt.rcParams["savefig.facecolor"] = SURFACE


def style_axes(ax, hide_y=False):
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRIDLINE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=13)
    if hide_y:
        ax.get_yaxis().set_visible(False)


# ============================================================
# 図1: スパイクの形、および「1日時点で予測が更新される」こと
# ============================================================
fig, ax = plt.subplots(figsize=(9.7, 4.6), dpi=200)

days = np.linspace(0, 30, 200)
normal_line = days / 30 * 100  # 単純な日割り（参考の点線）

# --- 実績（21日〜1日までは分かっている実際の積み上がり） ---
# 想定の基準ペース(スパイク61%→1日時点78%)より少し多く積み上がったケース
days_known = np.linspace(0, 11, 120)
actual_known = np.where(
    days_known < 0.3,
    days_known / 0.3 * 61,
    61 + (days_known - 0.3) / (11 - 0.3) * (85 - 61),
)

# --- ①21日時点の予測：基準カーブ（スパイク比率どおりに進むと仮定） ---
baseline_days = np.linspace(0, 30, 200)
baseline_curve = np.where(
    baseline_days < 0.3,
    baseline_days / 0.3 * 61,
    61 + (baseline_days - 0.3) / (30 - 0.3) * (100 - 61),
)

# --- ②1日時点の予測：実際の1日時点実績(85)をもとに更新した見立て ---
updated_days = np.linspace(11, 30, 100)
updated_curve = 85 + (updated_days - 11) / (30 - 11) * (109 - 85)

ax.plot(days, normal_line, color=INK_MUTED, linewidth=2, linestyle=":", label="通常商品（単純な日割りイメージ）")
ax.plot(baseline_days, baseline_curve, color=INK_MUTED, linewidth=2.5, linestyle="--",
        label="①21日時点の予測（想定どおり進むと仮定）")
ax.plot(days_known, actual_known, color=SERIES_BLUE, linewidth=3.5, label="実績（21日〜1日、分かっている実績）")
ax.plot(updated_days, updated_curve, color=SERIES_YELLOW, linewidth=3.5, linestyle="--",
        label="②1日時点の予測（実績を見て更新後）")

ax.annotate(
    "スパイク\n(先行計上分)",
    xy=(0.3, 61), xytext=(3.2, 24),
    fontsize=12.5, color=INK_PRIMARY,
    arrowprops=dict(arrowstyle="->", color=INK_SECONDARY, lw=1.5),
)
ax.annotate(
    "①21日時点の予測\n＝ 100",
    xy=(30, 100), xytext=(19.5, 87),
    fontsize=13, color=INK_PRIMARY, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=INK_MUTED, lw=1.5),
)
ax.annotate(
    "②1日時点の予測\n＝ 109（上方修正）",
    xy=(30, 109), xytext=(19.5, 113),
    fontsize=13, color=INK_PRIMARY, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="#c98500", lw=1.5),
)
ax.annotate(
    "想定より多めに\n積み上がっている",
    xy=(11, 85), xytext=(13.5, 62),
    fontsize=12, color=INK_SECONDARY,
    arrowprops=dict(arrowstyle="->", color=INK_SECONDARY, lw=1.2),
)

ax.set_xlim(0, 30)
ax.set_ylim(0, 122)
ax.set_xticks([0, 11, 30])
ax.set_xticklabels(["21日\n(サイクル開始)", "翌月1日", "翌月20日\n(月末)"], fontsize=13)
ax.set_yticks([0, 50, 100])
ax.set_yticklabels(["0%", "50%", "100%"], fontsize=13)
ax.set_ylabel("月末着地に対する\n累積実績の割合", fontsize=13, color=INK_SECONDARY)
style_axes(ax)
ax.legend(loc="lower right", fontsize=11, frameon=False)
ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)

fig.tight_layout()
fig.savefig("chart_spike_calendar.png")
plt.close(fig)


# ============================================================
# 図2: 進捗ゲージ（21日時点・1日時点・20日）
# ============================================================
fig, ax = plt.subplots(figsize=(9.5, 3.2), dpi=200)

stages = ["21日時点", "1日時点", "20日（月末）"]
values = [61, 78, 100]
colors = [ORDINAL_LIGHT, ORDINAL_MID, ORDINAL_DARK]
y_pos = np.arange(len(stages))[::-1]

for y, v, c, label in zip(y_pos, values, colors, stages):
    ax.barh(y, 100, color=GRIDLINE, height=0.5, zorder=1)
    ax.barh(y, v, color=c, height=0.5, zorder=2)
    ax.text(v + 2, y, f"約{v}%（約{round(v/10)}割）", va="center", fontsize=15, color=INK_PRIMARY)

ax.set_yticks(y_pos)
ax.set_yticklabels(stages, fontsize=15, color=INK_PRIMARY)
ax.set_xlim(0, 118)
ax.get_xaxis().set_visible(False)
for spine in ax.spines.values():
    spine.set_visible(False)

fig.tight_layout()
fig.savefig("chart_progress_gauge.png")
plt.close(fig)


# ============================================================
# 図3: 6月分の予測 vs 実績（統括部別）
# ============================================================
df = pd.read_csv("sample_sales.csv")
df_target = df[df["月分"] == 6]

start = pd.to_datetime(df["サイクル開始日"])
df = df.assign(経過日数_1日=(pd.to_datetime(df["1日時点日付"]) - start).dt.days)

# 別スクリプト(forecast_curve_example.py)のロジックで作った結果をそのまま埋め込み
summary = pd.DataFrame(
    {
        "実績（月末）": [5283, 6923, 6362, 6036, 9102],
        "予測（21日時点）": [5263, 6775, 6163, 5989, 8893],
        "予測（1日時点）": [5429, 7089, 6577, 6287, 9607],
    },
    index=["中部", "東日本", "西日本", "関西", "首都圏"],
)
summary = summary.sort_values("実績（月末）", ascending=False)

fig, ax = plt.subplots(figsize=(10.5, 4.6), dpi=200)
x = np.arange(len(summary))
width = 0.26

bars1 = ax.bar(x - width, summary["実績（月末）"], width, color=SERIES_BLUE, label="実績（月末）")
bars2 = ax.bar(x, summary["予測（21日時点）"], width, color=SERIES_AQUA, label="予測（21日時点）")
bars3 = ax.bar(x + width, summary["予測（1日時点）"], width, color=SERIES_YELLOW, label="予測（1日時点）")

for bars in (bars1, bars2, bars3):
    for b in bars:
        ax.text(
            b.get_x() + b.get_width() / 2, b.get_height() + 100,
            f"{int(b.get_height()):,}", ha="center", fontsize=10.5, color=INK_SECONDARY,
        )

ax.set_xticks(x)
ax.set_xticklabels(summary.index, fontsize=14, color=INK_PRIMARY)
ax.get_yaxis().set_visible(False)
style_axes(ax)
ax.legend(loc="upper right", fontsize=12, frameon=False, ncol=3, bbox_to_anchor=(1.0, 1.15))

fig.tight_layout()
fig.savefig("chart_comparison_bar.png")
plt.close(fig)

print("charts generated: chart_spike_calendar.png, chart_progress_gauge.png, chart_comparison_bar.png")
