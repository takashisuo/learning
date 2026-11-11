"""加法性: ベースライン + ΣSHAP = 予測値(モデルスケール・元スケール)と、Ridge による実装検証。"""
import numpy as np

from src.explain import group_decomposition_table, ridge_check, ridge_explanation
from src.xai_platform.explain import DEFAULT_ADDITIVITY_TOL as TOL


def test_additivity_model_scale(analysis):
    ex, X = analysis.ex, analysis.X_all
    err = np.abs(ex.base_raw + ex.raw(X).values.sum(axis=1) - ex.model.predict_raw(X))
    assert err.max() < TOL


def test_additivity_original_scale(analysis):
    ex, X = analysis.ex, analysis.X_all
    sv = ex(X)
    err = np.abs(sv.base_values + sv.values.sum(axis=1) - ex.model.predict(X))
    assert err.max() < TOL


def test_baseline_is_mean_training_prediction(analysis):
    """ベースライン(E[f] + 訓練平均roll7)は訓練期間の平均予測に一致する(行サンプリングなし)。"""
    mean_pred = analysis.model.predict(analysis.X_train).mean()
    assert abs(analysis.ex.base - mean_pred) < 1e-3


def test_group_contributions_sum_to_prediction(analysis):
    g = analysis.gdf
    assert np.allclose(analysis.ex.base + g.sum(axis=1), analysis.pred_all, atol=TOL)


def test_decomposition_table_closes(analysis):
    for d in analysis.days()["日付"]:
        row = analysis._row(d)
        tbl = group_decomposition_table(row, analysis.groups, analysis.unit,
                                        pred=analysis.pred_all[d])
        parts = tbl["寄与合計"].iloc[1:-1].sum()
        assert abs(tbl["寄与合計"].iloc[0] + parts - analysis.pred_all[d]) < TOL


def test_ridge_shap_equals_coef_times_centered_x(analysis):
    ridge = analysis._model("ridge", analysis.tf)
    rc = ridge_check(ridge, analysis.X_train, analysis.X_test)
    assert rc.iloc[:, -1].max() < TOL
    assert rc.attrs["ベースライン誤差"] < TOL


def test_ridge_original_scale_additivity(analysis):
    ridge = analysis._model("ridge", analysis.tf)
    X = analysis.X_test
    sv, base = ridge_explanation(ridge, analysis.X_train, X, analysis.ex.level_ref,
                                 analysis.cfg["level_name"])
    assert np.allclose(base + sv.values.sum(axis=1), ridge.predict(X), atol=TOL)
