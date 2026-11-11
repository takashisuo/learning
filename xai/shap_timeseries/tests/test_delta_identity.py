"""Δ恒等式: Δ予測値 = ΣΔSHAP(比較時点との差・What-if シナリオの前後)。"""
import numpy as np
import pandas as pd
import pytest

from src.decompose import delta_decomposition
from src.whatif import WhatIf, parse_scenarios
from src.xai_platform.explain import DEFAULT_ADDITIVITY_TOL as TOL


@pytest.fixture(scope="module")
def wi(analysis):
    wc = analysis.cfg["whatif"]
    return WhatIf(analysis.ex, analysis.X_train, analysis.fs.exog,
                  tuple(wc["bounds_quantiles"]), wc["correlated_pairs"])


@pytest.mark.parametrize("lag", [1, 7, 364])
def test_delta_identity_between_dates(analysis, lag):
    d = delta_decomposition(analysis.ts, analysis.group_names, lag, analysis.X_test.index)
    assert len(d) == len(analysis.X_test)
    assert d["恒等式の誤差"].max() < TOL


def test_delta_identity_feature_level(analysis):
    """グループ集約前(特徴量単位)でも ΔΣSHAP = Δ予測値。"""
    sv = analysis.sv_all.values
    pred = analysis.pred_all.to_numpy()
    assert np.allclose(np.diff(pred), np.diff(sv.sum(axis=1)), atol=TOL)


def test_whatif_day_identity(analysis, wi):
    scen = {**parse_scenarios(analysis.cfg["whatif"]["single"], analysis.fs.exog),
            **parse_scenarios(analysis.cfg["whatif"]["multi"], analysis.fs.exog)}
    for d in analysis.days()["日付"]:
        summary, dshap, _ = wi.day(analysis.X_all.loc[d], scen)
        assert np.allclose(summary["Δ予測値(同時)"], dshap.sum(axis=1).to_numpy(), atol=TOL)


def test_whatif_period_identity_and_ar_untouched(analysis, wi):
    scen = parse_scenarios(analysis.cfg["whatif"]["period"], analysis.fs.exog)
    X = analysis.X_test
    for changes in scen.values():
        new, _ = wi.apply_frame(X, changes)
        untouched = [c for c in X.columns if c not in changes]
        pd.testing.assert_frame_equal(new[untouched], X[untouched])
        d = analysis.model.predict(new) - analysis.model.predict(X)
        ds = analysis.ex.raw(new).values - analysis.ex.raw(X).values
        assert np.allclose(d, ds.sum(axis=1), atol=TOL)


def test_whatif_rejects_non_exogenous(analysis):
    with pytest.raises(ValueError):
        parse_scenarios({"NG": {"lag_1": ["abs", 100]}}, analysis.fs.exog)


def test_whatif_respects_bounds(analysis, wi):
    X = analysis.X_test
    new, clipped = wi.apply_frame(X, {"temp_c": ("abs", 50.0)})
    assert (new["temp_c"] <= wi.bounds.loc["temp_c"].iloc[1] + 1e-12).all()
    assert clipped.all()
