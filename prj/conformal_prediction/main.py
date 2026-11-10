"""時系列データ + 機械学習モデル + Split Conformal Prediction のサンプル。

Prophet は自動で信頼区間を計算してくれるが、scikit-learn 等の一般的な機械学習
モデルにはその機能がない。ここでは Split Conformal Prediction を用いて、
機械学習モデルの点予測に対して疑似的な予測区間(信頼区間相当)を付与する例を示す。

実行方法:
    python main.py
実行すると output/conformal_prediction.html に可視化結果が出力される。
"""

from pathlib import Path

import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor

from conformal import SplitConformalRegressor
from data import create_features, generate_sample_timeseries
from models import TrendResidualRegressor

ALPHA = 0.1  # 有意水準 -> 目標被覆率 90%
FEATURE_EXCLUDE_COLS = {"date", "y"}


def time_series_split(df, train_ratio=0.6, calib_ratio=0.2):
    """時系列順序を保ったまま train / calibration / test に分割する。"""
    n = len(df)
    train_end = int(n * train_ratio)
    calib_end = int(n * (train_ratio + calib_ratio))

    train_df = df.iloc[:train_end]
    calib_df = df.iloc[train_end:calib_end]
    test_df = df.iloc[calib_end:]
    return train_df, calib_df, test_df


def visualize(train_df, calib_df, test_df, y_pred, lower, upper, coverage, out_path):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=train_df["date"], y=train_df["y"],
        mode="lines", name="実測値 (train)",
        line=dict(color="#888888", width=1),
    ))
    fig.add_trace(go.Scatter(
        x=calib_df["date"], y=calib_df["y"],
        mode="lines", name="実測値 (calibration)",
        line=dict(color="#4c78a8", width=1),
    ))
    fig.add_trace(go.Scatter(
        x=test_df["date"], y=test_df["y"],
        mode="lines", name="実測値 (test)",
        line=dict(color="#111111", width=2),
    ))

    # 予測区間(帯)
    fig.add_trace(go.Scatter(
        x=list(test_df["date"]) + list(test_df["date"])[::-1],
        y=list(upper) + list(lower)[::-1],
        fill="toself",
        fillcolor="rgba(214, 39, 40, 0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        name=f"予測区間 ({int((1 - ALPHA) * 100)}%)",
    ))

    # 点予測
    fig.add_trace(go.Scatter(
        x=test_df["date"], y=y_pred,
        mode="lines", name="点予測 (test)",
        line=dict(color="#d62728", width=2, dash="dot"),
    ))

    for boundary_date in (calib_df["date"].iloc[0], test_df["date"].iloc[0]):
        fig.add_vline(x=boundary_date, line_width=1, line_dash="dash", line_color="gray")

    fig.update_layout(
        title=(
            "Conformal Prediction による予測区間<br>"
            f"<sub>目標被覆率: {int((1 - ALPHA) * 100)}% / "
            f"実測被覆率: {coverage * 100:.1f}% (test, n={len(test_df)})</sub>"
        ),
        xaxis_title="date",
        yaxis_title="y",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path))
    print(f"可視化結果を出力しました: {out_path.resolve()}")


def main():
    raw_df = generate_sample_timeseries(n_days=730)
    feat_df = create_features(raw_df)

    train_df, calib_df, test_df = time_series_split(feat_df)
    feature_cols = [c for c in feat_df.columns if c not in FEATURE_EXCLUDE_COLS]

    X_train, y_train = train_df[feature_cols], train_df["y"]
    X_calib, y_calib = calib_df[feature_cols], calib_df["y"]
    X_test, y_test = test_df[feature_cols], test_df["y"]

    base_model = TrendResidualRegressor(
        GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42,
        ),
        trend_col="day_index",
    )
    conformal_model = SplitConformalRegressor(base_model, alpha=ALPHA)
    conformal_model.fit(X_train, y_train)
    conformal_model.calibrate(X_calib, y_calib)

    y_pred, lower, upper = conformal_model.predict(X_test)
    coverage = conformal_model.coverage(y_test, lower, upper)

    print(f"キャリブレーションから求めた区間半幅 q_hat = {conformal_model.q_hat:.3f}")
    print(f"目標被覆率: {(1 - ALPHA) * 100:.1f}%")
    print(f"実測被覆率 (test): {coverage * 100:.1f}% (n={len(test_df)})")

    out_path = Path(__file__).parent / "output" / "conformal_prediction.html"
    visualize(train_df, calib_df, test_df, y_pred, lower, upper, coverage, out_path)


if __name__ == "__main__":
    main()
