"""Δ恒等式: Δ予測値 = ΣΔSHAP(比較時点との差・What-if シナリオの前後)。"""
import numpy as np
import pandas as pd
import pytest

from src.decompose import delta_decomposition
from src.whatif import parse_scenarios
from src.xai_platform._core.whatif import apply_changes_frame

TOL = 1e-6


@pytest.mark.parametrize("lag", [1, 7, 364])
def test_delta_identity_between_dates(analysis, lag):
    d = delta_decomposition(analysis.az, lag, analysis.X_test.index)
    assert len(d) == len(analysis.X_test)
    assert d["恒等式の誤差"].max() < TOL


def test_delta_identity_feature_level(analysis):
    """グループ集約前(特徴量単位)でも ΔΣSHAP = Δ予測値。"""
    az = analysis.az
    assert np.allclose(np.diff(az.pred.to_numpy()), np.diff(az.sv.values.sum(axis=1)), atol=TOL)


def test_whatif_day_identity(analysis):
    az = analysis.az
    scen = {**parse_scenarios(analysis.cfg["whatif"]["single"], analysis.fs.exog),
            **parse_scenarios(analysis.cfg["whatif"]["multi"], analysis.fs.exog)}
    for d in analysis.days()["日付"]:
        r = az.whatif(d, scen)
        assert np.allclose(r.tables["シナリオ別の結果"]["Δ予測値(同時)"],
                           r.tables["ΔSHAP"].sum(axis=1).to_numpy(), atol=TOL)


def test_whatif_period_identity_and_ar_untouched(analysis):
    az = analysis.az
    scen = parse_scenarios(analysis.cfg["whatif"]["period"], analysis.fs.exog)
    X = analysis.X_test
    for changes in scen.values():
        new, _ = apply_changes_frame(X, changes, az.sd, az.bounds)
        untouched = [c for c in X.columns if c not in changes]
        pd.testing.assert_frame_equal(new[untouched], X[untouched])
        d = az.predict(new) - az.predict(X)
        ds = az.shap_raw(new).values - az.shap_raw(X).values
        assert np.allclose(d, ds.sum(axis=1), atol=TOL)
    az.whatif_period(scen, rows=X.index)  # 内部で Δ予測値 = ΣΔSHAP を確認する


def test_whatif_rejects_non_exogenous(analysis):
    with pytest.raises(ValueError):
        parse_scenarios({"NG": {"lag_1": ["abs", 100]}}, analysis.fs.exog)


def test_facade_rejects_offset_features(analysis):
    """補正項が参照する roll7_mean を動かすシナリオは Facade が拒否する。"""
    d = analysis.X_test.index[0]
    with pytest.raises(ValueError):
        analysis.az.whatif(d, {"NG": {"roll7_mean": ("abs", 100.0)}})


def test_whatif_respects_bounds(analysis):
    az, X = analysis.az, analysis.X_test
    new, clipped = apply_changes_frame(X, {"temp_c": ("abs", 50.0)}, az.sd, az.bounds)
    assert (new["temp_c"] <= az.bounds.loc["temp_c"].iloc[1] + 1e-12).all()
    assert clipped.all()
