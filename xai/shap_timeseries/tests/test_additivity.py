"""加法性: ベースライン + ΣSHAP = 予測値(モデル出力・予測値)と、Ridge による実装検証。"""
import numpy as np

from src.explain import ridge_check, ridge_explanation

TOL = 1e-6


def test_facade_predict_matches_model(analysis):
    """ShapAnalyzer の予測値(モデル出力 + 補正項)が TSModel の予測値と一致する。"""
    X = analysis.fs.X
    assert np.allclose(analysis.az.predict(X), analysis.model.predict(X), atol=TOL)


def test_additivity_model_output(analysis):
    az, X = analysis.az, analysis.fs.X
    err = np.abs(az.base_raw + az.shap_raw(X).values.sum(axis=1) - az.predict_raw(X))
    assert err.max() < TOL


def test_additivity_prediction(analysis):
    az, X = analysis.az, analysis.fs.X
    sv = az.shap(X)
    err = np.abs(sv.base_values + sv.values.sum(axis=1) - az.predict(X))
    assert err.max() < TOL


def test_baseline_is_mean_training_prediction(analysis):
    """ベースライン(E[f] + 訓練平均roll7)は訓練期間の平均予測に一致する(行サンプリングなし)。"""
    mean_pred = analysis.model.predict(analysis.X_train).mean()
    assert abs(analysis.az.base - mean_pred) < 1e-3


def test_group_contributions_sum_to_prediction(analysis):
    az = analysis.az
    assert np.allclose(az.base + az.contributions().sum(axis=1), az.pred, atol=TOL)


def test_decomposition_table_closes(analysis):
    az = analysis.az
    for d in analysis.days()["日付"]:
        tbl = az.decompose(d).tables["分解表"]
        groups = tbl[tbl["グループ"].isin(az.group_names)]["寄与合計"].sum()
        assert abs(tbl["寄与合計"].iloc[0] + groups - az.pred[d]) < TOL


def test_ridge_shap_equals_coef_times_centered_x(analysis):
    ridge = analysis._model("ridge", analysis.tf)
    rc = ridge_check(ridge, analysis.X_train, analysis.X_test)
    assert rc.iloc[:, -1].max() < TOL
    assert rc.attrs["ベースライン誤差"] < TOL


def test_ridge_original_scale_additivity(analysis):
    ridge = analysis._model("ridge", analysis.tf)
    X = analysis.X_test
    sv, base = ridge_explanation(ridge, analysis.X_train, X, analysis.az.offset_ref,
                                 analysis.cfg["level_name"])
    assert np.allclose(base + sv.values.sum(axis=1), ridge.predict(X), atol=TOL)
