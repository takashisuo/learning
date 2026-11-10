"""自作 Conformal Prediction と MAPIE ライブラリの比較サンプル。

MAPIE (https://mapie.readthedocs.io/) には主に2系統の回帰用クラスがある:

- `SplitConformalRegressor`  : 通常のクロスセクショナル(i.i.d.に近い)データ向け
- `TimeSeriesRegressor`      : 時系列データ向け(method="enbpi" で EnbPI を使用)

親フォルダ (`..`) で自作した `SplitConformalRegressor` (`conformal.py`) と
全く同じデータ・特徴量・ベースモデル (`TrendResidualRegressor`) を使い、
被覆率(coverage)と区間幅(width)を並べて比較する。

実行方法:
    python main.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from mapie.regression import SplitConformalRegressor as MapieSplitConformalRegressor
from mapie.regression import TimeSeriesRegressor as MapieTimeSeriesRegressor
from mapie.subsample import BlockBootstrap
from sklearn.ensemble import GradientBoostingRegressor

# 親フォルダ (conformal_prediction/) のモジュールを再利用する
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conformal import SplitConformalRegressor as ScratchSplitConformalRegressor  # noqa: E402
from data import create_features, generate_sample_timeseries  # noqa: E402
from models import TrendResidualRegressor  # noqa: E402
from main import FEATURE_EXCLUDE_COLS, time_series_split  # noqa: E402

# MAPIE 内部で prefit 済みモデルの状態チェックが誤検知して出す警告を抑制
warnings.filterwarnings("ignore", message="Estimator does not appear fitted")

ALPHA = 0.1
CONFIDENCE_LEVEL = 1 - ALPHA


def build_base_model() -> TrendResidualRegressor:
    """3手法とも同じベースモデルを使うことで公平に比較する。"""
    return TrendResidualRegressor(
        GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42,
        ),
        trend_col="day_index",
    )


def coverage_and_width(y_true, lower, upper) -> tuple[float, float]:
    y_true = np.asarray(y_true)
    coverage = float(np.mean((y_true >= lower) & (y_true <= upper)))
    width = float(np.mean(upper - lower))
    return coverage, width


def run_scratch(X_train, y_train, X_calib, y_calib, X_test):
    """自作の SplitConformalRegressor(conformal.py)。"""
    model = ScratchSplitConformalRegressor(build_base_model(), alpha=ALPHA)
    model.fit(X_train, y_train)
    model.calibrate(X_calib, y_calib)
    y_pred, lower, upper = model.predict(X_test)
    return y_pred, lower, upper


def run_mapie_split(X_train, y_train, X_calib, y_calib, X_test):
    """MAPIE の SplitConformalRegressor(クロスセクショナル/i.i.d.向け)。"""
    base_model = build_base_model()
    base_model.fit(X_train, y_train)

    model = MapieSplitConformalRegressor(
        estimator=base_model, confidence_level=CONFIDENCE_LEVEL, prefit=True,
    )
    model.conformalize(X_calib, y_calib)
    y_pred, y_pis = model.predict_interval(X_test)
    return y_pred, y_pis[:, 0, 0], y_pis[:, 1, 0]


def run_mapie_enbpi(X_train, y_train, X_calib, y_calib, X_test):
    """MAPIE の TimeSeriesRegressor(method="enbpi", 時系列向け)。

    EnbPI はブートストラップで学習した複数モデルのアンサンブルから
    Out-of-Bag 残差を使って区間を求めるため、calibration 用データを
    別に分けず学習データに含める。
    """
    X_fit = pd.concat([X_train, X_calib])
    y_fit = pd.concat([y_train, y_calib])

    cv = BlockBootstrap(n_resamplings=30, length=50, overlapping=True, random_state=42)
    model = MapieTimeSeriesRegressor(
        estimator=build_base_model(), method="enbpi", cv=cv, agg_function="mean", n_jobs=-1,
    )
    model.fit(X_fit, y_fit)
    y_pred, y_pis = model.predict(X_test, ensemble=True, confidence_level=CONFIDENCE_LEVEL)
    return y_pred, y_pis[:, 0, 0], y_pis[:, 1, 0]


def visualize(test_df, results: dict, out_path: Path):
    colors = {
        "自作 SplitConformalRegressor": "31, 119, 180",
        "MAPIE SplitConformalRegressor": "255, 127, 14",
        "MAPIE TimeSeriesRegressor (EnbPI)": "44, 160, 44",
    }

    # 3手法を縦に並べて独立描画しつつ、比較しやすいよう y軸レンジは全パネル共通にする
    all_lower = np.concatenate([lower for _, lower, _ in results.values()])
    all_upper = np.concatenate([upper for _, _, upper in results.values()])
    y_min = min(test_df["y"].min(), all_lower.min())
    y_max = max(test_df["y"].max(), all_upper.max())
    margin = (y_max - y_min) * 0.05
    y_range = [y_min - margin, y_max + margin]

    fig = make_subplots(
        rows=len(results), cols=1,
        shared_xaxes=True,
        subplot_titles=list(results.keys()),
        vertical_spacing=0.08,
    )

    for row, (name, (y_pred, lower, upper)) in enumerate(results.items(), start=1):
        rgb = colors[name]
        fig.add_trace(go.Scatter(
            x=test_df["date"], y=test_df["y"],
            mode="lines", name="実測値 (test)",
            line=dict(color="#111111", width=2),
            legendgroup="actual", showlegend=(row == 1),
        ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=list(test_df["date"]) + list(test_df["date"])[::-1],
            y=list(upper) + list(lower)[::-1],
            fill="toself",
            fillcolor=f"rgba({rgb}, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name=f"{name} 区間",
        ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=test_df["date"], y=y_pred,
            mode="lines", name=f"{name} 点予測",
            line=dict(color=f"rgb({rgb})", width=1.5, dash="dot"),
        ), row=row, col=1)
        fig.update_yaxes(range=y_range, title_text="y", row=row, col=1)

    fig.update_xaxes(title_text="date", row=len(results), col=1)
    fig.update_layout(
        title=f"自作 vs MAPIE の Conformal Prediction 比較(目標被覆率 {int(CONFIDENCE_LEVEL * 100)}%)",
        template="plotly_white",
        height=350 * len(results),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path))
    print(f"\n可視化結果を出力しました: {out_path.resolve()}")


def main():
    raw_df = generate_sample_timeseries(n_days=730)
    feat_df = create_features(raw_df)
    train_df, calib_df, test_df = time_series_split(feat_df)
    feature_cols = [c for c in feat_df.columns if c not in FEATURE_EXCLUDE_COLS]

    X_train, y_train = train_df[feature_cols], train_df["y"]
    X_calib, y_calib = calib_df[feature_cols], calib_df["y"]
    X_test, y_test = test_df[feature_cols], test_df["y"]

    runners = {
        "自作 SplitConformalRegressor": run_scratch,
        "MAPIE SplitConformalRegressor": run_mapie_split,
        "MAPIE TimeSeriesRegressor (EnbPI)": run_mapie_enbpi,
    }

    results = {}
    print(f"目標被覆率: {CONFIDENCE_LEVEL * 100:.0f}%  (test, n={len(test_df)})\n")
    print(f"{'手法':38s}{'被覆率':>10s}{'平均区間幅':>12s}")
    for name, runner in runners.items():
        y_pred, lower, upper = runner(X_train, y_train, X_calib, y_calib, X_test)
        results[name] = (y_pred, lower, upper)
        coverage, width = coverage_and_width(y_test, lower, upper)
        print(f"{name:38s}{coverage * 100:9.1f}%{width:12.2f}")

    out_path = Path(__file__).parent / "output" / "mapie_comparison.html"
    visualize(test_df, results, out_path)


if __name__ == "__main__":
    main()
