"""加法性・Ridge検証・What-if分解の自動テスト。"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data, explain as ex, model as mdl, whatif as wi  # noqa: E402


@pytest.fixture(scope="module")
def ds():
    return data.load_california()


@pytest.fixture(scope="module")
def lgbm(ds):
    return mdl.train_lgbm(ds)


@pytest.fixture(scope="module")
def explainer(lgbm):
    return ex.make_tree_explainer(lgbm)


def test_baseline_is_train_mean_prediction(ds, lgbm, explainer):
    assert abs(ex.baseline(explainer) - lgbm.predict(ds.X_train).mean()) < 1e-6


def test_additivity_all_test_samples(ds, lgbm, explainer):
    sv = ex.compute_shap(explainer, ds.X_test)
    check = ex.additivity_check(explainer, lgbm, ds.X_test, sv)
    assert len(check) == len(ds.X_test)
    assert check["合格"].all(), f"最大誤差 {check['絶対誤差'].max():.3e}"


def test_ridge_shap_equals_coef_times_centered_x(ds):
    ridge = mdl.train_ridge(ds)
    chk = ex.ridge_shap_check(ridge, ds.X_train, ds.X_test)
    assert chk.iloc[:, 1].max() < 1e-8
    assert chk.attrs["ベースライン誤差"] < 1e-8


def test_additivity_diabetes():
    dsd = data.load_diabetes_dataset()
    m = mdl.train_lgbm(dsd, n_estimators=200, num_leaves=15)
    e = ex.make_tree_explainer(m)
    sv = ex.compute_shap(e, dsd.X_test)
    assert ex.additivity_check(e, m, dsd.X_test, sv)["合格"].all()


def test_whatif_dshap_sums_to_delta_prediction(ds, lgbm, explainer):
    sd, bounds = wi.robust_sd(ds.X_train), wi.feature_bounds(ds.X_train)
    x = ds.X_test.iloc[0]
    summ, dshap, _ = wi.scenario_analysis(lgbm, explainer, x, sd, bounds)
    np.testing.assert_allclose(dshap.sum(axis=1).values, summ["Δ予測値(同時)"].values, atol=1e-6)
    np.testing.assert_allclose(summ["相互作用分(同時−単独合計)"],
                               summ["Δ予測値(同時)"] - summ["単独Δの合計"], atol=1e-12)


def test_whatif_clips_to_training_range(ds):
    sd, bounds = wi.robust_sd(ds.X_train), wi.feature_bounds(ds.X_train)
    x = ds.X_test.iloc[0].copy()
    x["HouseAge"] = bounds.loc["HouseAge", "下限(1%)"]
    new, log = wi.apply_changes(x, {"HouseAge": ("abs", -10.0)}, sd, bounds)
    assert new["HouseAge"] == bounds.loc["HouseAge", "下限(1%)"]
    assert log[0]["外挿警告"]


def test_group_definition_covers_all_features(ds, explainer):
    sv = ex.compute_shap(explainer, ds.X_test.iloc[:50])
    grp = ex.group_contributions(sv)
    np.testing.assert_allclose(grp.sum(axis=1).values, sv.values.sum(axis=1), atol=1e-12)
