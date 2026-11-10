"""AirPassengers データで Prophet と MAPIE TimeSeriesRegressor(EnbPI)を比較する。

データ: https://www.salesanalytics.co.jp/datascience/datascience085/ で紹介されている
月次航空旅客数データ(1949-01 〜 1960-12)。

やること:
1. 直近24ヶ月をテスト期間として、それ以外を学習データにする
2. Prophet で学習・予測(信頼区間は自動算出)
3. MAPIE の TimeSeriesRegressor(method="enbpi")で学習・予測(予測区間をConformal Predictionで算出)
4. 被覆率・区間幅を比較し、plotlyで並べて可視化する

実行方法:
    python main.py
"""

import os
from pathlib import Path

# --- Windows環境でProphet(cmdstanpy)がtbb.dllを見つけられない問題への対処 ---
import prophet

tbb_dirs = list((Path(prophet.__file__).parent / "stan_model").glob("cmdstan-*/stan/lib/stan_math/lib/tbb"))
if tbb_dirs:
    os.environ["PATH"] = str(tbb_dirs[0]) + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from mapie.regression import TimeSeriesRegressor
from mapie.subsample import BlockBootstrap

CONFIDENCE_LEVEL = 0.9
N_TEST = 24  # 直近24ヶ月をテスト期間とする

# --- データ読み込み ---
df = pd.read_csv(Path(__file__).parent / "AirPassengers.csv")
df["ds"] = pd.to_datetime(df["Month"])
df["y"] = df["#Passengers"]
df = df[["ds", "y"]]

train_df = df.iloc[:-N_TEST]
test_df = df.iloc[-N_TEST:]
y_true = test_df["y"].values

# --- 1. Prophet ---
prophet_model = Prophet(interval_width=CONFIDENCE_LEVEL)
prophet_model.fit(train_df)
prophet_forecast = prophet_model.predict(test_df[["ds"]])
prophet_pred = prophet_forecast["yhat"].values
prophet_lower = prophet_forecast["yhat_lower"].values
prophet_upper = prophet_forecast["yhat_upper"].values

# --- 2. MAPIE TimeSeriesRegressor (EnbPI) ---
# ラグ特徴量(前月・前々月・3ヶ月前・去年同月)と月を特徴量にする
feat_df = df.copy()
feat_df["month"] = feat_df["ds"].dt.month
for lag in (1, 2, 3, 12):
    feat_df[f"lag_{lag}"] = feat_df["y"].shift(lag)
feat_df = feat_df.dropna().reset_index(drop=True)

feature_cols = ["month", "lag_1", "lag_2", "lag_3", "lag_12"]
mapie_train = feat_df.iloc[:-N_TEST]
mapie_test = feat_df.iloc[-N_TEST:]

cv = BlockBootstrap(n_resamplings=30, length=12, overlapping=True, random_state=42)
mapie_model = TimeSeriesRegressor(
    estimator=LinearRegression(), method="enbpi", cv=cv, agg_function="mean", n_jobs=-1,
)
mapie_model.fit(mapie_train[feature_cols], mapie_train["y"])
mapie_pred, mapie_pis = mapie_model.predict(
    mapie_test[feature_cols], ensemble=True, confidence_level=CONFIDENCE_LEVEL,
)
mapie_lower = mapie_pis[:, 0, 0]
mapie_upper = mapie_pis[:, 1, 0]

# --- 3. 被覆率・区間幅を比較 ---
def coverage_and_width(y_true, lower, upper):
    coverage = np.mean((y_true >= lower) & (y_true <= upper))
    width = np.mean(upper - lower)
    return coverage, width

prophet_coverage, prophet_width = coverage_and_width(y_true, prophet_lower, prophet_upper)
mapie_coverage, mapie_width = coverage_and_width(y_true, mapie_lower, mapie_upper)

print(f"目標被覆率: {CONFIDENCE_LEVEL * 100:.0f}%  (test, n={N_TEST})\n")
print(f"{'手法':30s}{'被覆率':>10s}{'平均区間幅':>12s}")
print(f"{'Prophet':30s}{prophet_coverage * 100:9.1f}%{prophet_width:12.2f}")
print(f"{'MAPIE TimeSeriesRegressor':30s}{mapie_coverage * 100:9.1f}%{mapie_width:12.2f}")

# --- 4. 可視化(実測値・点予測・予測区間を2手法並べて表示、y軸レンジは共通) ---
y_min = min(df["y"].min(), mapie_lower.min(), prophet_lower.min())
y_max = max(df["y"].max(), mapie_upper.max(), prophet_upper.max())
margin = (y_max - y_min) * 0.05
y_range = [y_min - margin, y_max + margin]

test_start = test_df["ds"].iloc[0]

fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    subplot_titles=("Prophet", "MAPIE TimeSeriesRegressor (EnbPI)"),
    vertical_spacing=0.1,
)

for row, (name, pred, lower, upper) in enumerate(
    [("Prophet", prophet_pred, prophet_lower, prophet_upper),
     ("MAPIE", mapie_pred, mapie_lower, mapie_upper)],
    start=1,
):
    # 実測値は全期間(学習+テスト)を表示し、どこからテスト期間かを一目で分かるようにする
    fig.add_trace(go.Scatter(
        x=df["ds"], y=df["y"],
        mode="lines", name="実測値(全期間)",
        line=dict(color="#111111", width=2),
        legendgroup="actual", showlegend=(row == 1),
    ), row=row, col=1)
    fig.add_trace(go.Scatter(
        x=list(test_df["ds"]) + list(test_df["ds"])[::-1],
        y=list(upper) + list(lower)[::-1],
        fill="toself",
        fillcolor="rgba(31, 119, 180, 0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        name=f"{name} 予測区間",
    ), row=row, col=1)
    fig.add_trace(go.Scatter(
        x=test_df["ds"], y=pred,
        mode="lines", name=f"{name} 点予測",
        line=dict(color="rgb(31, 119, 180)", width=1.5, dash="dot"),
    ), row=row, col=1)
    fig.update_yaxes(range=y_range, title_text="#Passengers", row=row, col=1)
    fig.add_vline(
        x=test_start.to_pydatetime(), line_width=1, line_dash="dash", line_color="gray",
        row=row, col=1,
    )

fig.add_annotation(
    x=test_start.to_pydatetime(), y=y_range[1], row=1, col=1,
    text="予測開始", showarrow=False, yshift=10, font=dict(color="gray"),
)

fig.update_xaxes(title_text="Month", row=2, col=1)
fig.update_layout(
    title=f"AirPassengers: Prophet vs MAPIE TimeSeriesRegressor(目標被覆率 {int(CONFIDENCE_LEVEL * 100)}%)",
    template="plotly_white",
    height=700,
    legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0),
)

out_path = Path(__file__).parent / "output" / "airpassengers_comparison.html"
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.write_html(str(out_path))
print(f"\n可視化結果を出力しました: {out_path.resolve()}")
