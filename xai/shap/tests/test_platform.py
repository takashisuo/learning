"""xai_platform がデータセットに依存せず使えることを、人工データで確認する。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.xai_platform import Unit, explain as ex, plots as pl, whatif as wi  # noqa: E402
from src.xai_platform.data import split_dataset  # noqa: E402
from src.xai_platform.io import OutputDirs  # noqa: E402
from src.xai_platform.model import evaluation_table, fit  # noqa: E402


@pytest.fixture(scope="module")
def setup():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(600, 4)), columns=["a", "b", "c", "d"])
    y = pd.Series(2 * X["a"] + X["b"] * X["c"] + 0.1 * rng.normal(size=600), name="y")
    ds = split_dataset(X, y, "synthetic", 0.25, 0)
    model = fit(LGBMRegressor(n_estimators=100, random_state=0, verbose=-1), ds)
    explainer = ex.make_tree_explainer(model)
    return ds, model, explainer, ex.compute_shap(explainer, ds.X_test)


def test_additivity_and_eval(setup):
    ds, model, explainer, sv = setup
    assert ex.additivity_check(explainer, model, ds.X_test, sv)["合格"].all()
    assert evaluation_table({"m": model}, ds).loc["m", "R2"] > 0.5


def test_linear_check(setup):
    ds = setup[0]
    chk = ex.linear_shap_check(fit(Ridge(), ds), ds.X_train, ds.X_test)
    assert chk.iloc[:, 1].max() < 1e-8


def test_unit_without_alt_has_no_alt_column(setup):
    _, _, explainer, sv = setup
    tbl = ex.decomposition_table(sv[0], ex.baseline(explainer), Unit("点"))
    assert list(tbl.columns) == ["特徴量", "現在の値", "SHAP値", "累積"]
    assert np.isclose(tbl["累積"].iloc[-1], ex.baseline(explainer) + sv[0].values.sum())
    assert "約" not in ex.summary_sentence(sv[0], ex.baseline(explainer), Unit("点"))


def test_unit_with_alt():
    u = Unit("千円", "円", 1000)
    assert u.signed(-0.5) == "−0.500(約−500.0円)"
    assert u.alt_column("SHAP値") == "SHAP値(円)"


def test_interaction_detects_bc_pair(setup):
    ds, _, explainer, _ = setup
    inter, _ = ex.interaction_values(explainer, ds.X_test, n=100, seed=0)
    top = ex.interaction_strength(inter, ds.feature_names).iloc[0]
    assert {top["特徴量1"], top["特徴量2"]} == {"b", "c"}


def test_whatif_generic(setup):
    ds, model, explainer, _ = setup
    sd, bounds = wi.robust_sd(ds.X_train, 0.05, 0.95), wi.feature_bounds(ds.X_train, 0.05, 0.95)
    assert list(bounds.columns) == ["下限(5%)", "上限(95%)"]
    x = ds.X_test.iloc[0]
    summ, dshap, _ = wi.scenario_analysis(model, explainer, x, sd, bounds,
                                          {"T1 bとcを上げる": {"b": ("sd", 1), "c": ("sd", 1)}})
    np.testing.assert_allclose(dshap.sum(axis=1).values, summ["Δ予測値(同時)"].values, atol=1e-6)
    one = wi.single_variable_whatif(model, x, sd, bounds, exclude=["d"])
    assert set(one["特徴量"]) == {"a", "b", "c"}


def test_provenance_records_interfaces(setup, tmp_path):
    from src.xai_platform.io import save_csv
    from src.xai_platform.provenance import Manifest, MANIFEST

    ds, model, explainer, sv = setup
    out = OutputDirs(tmp_path).ensure()
    MANIFEST.reset()
    with MANIFEST.step("テスト"):
        save_csv(ex.dependence_table(sv, "a"), out.table("dep.csv"), index=False)
        summ, _, _ = wi.scenario_analysis(model, explainer, ds.X_test.iloc[0],
                                          wi.robust_sd(ds.X_train), wi.feature_bounds(ds.X_train),
                                          {"T1 a": {"a": ("abs", 0.1)}})
        save_csv(summ, out.table("sc.csv"))
        pl.beeswarm(sv, out.figure("b.png"))
    df = MANIFEST.to_frame(tmp_path).set_index("ファイル")
    assert df.loc["tables/dep.csv", "生成I/F"] == "xai_platform.explain.dependence_table"
    assert df.loc["tables/sc.csv", "生成I/F"] == "xai_platform.whatif.scenario_analysis"
    assert df.loc["figures/b.png", "SHAP描画関数"] == "shap.plots.beeswarm"
    assert (df["分析ステップ"] == "テスト").all()
    csv, md = MANIFEST.save(tmp_path)
    assert csv.exists() and md.exists()
    MANIFEST.reset()


def test_plots_write_files(setup, tmp_path):
    ds, model, explainer, sv = setup
    out = OutputDirs(tmp_path).ensure()
    base = ex.baseline(explainer)
    grp = ex.group_contributions(sv, {"G1": ["a"], "G2": ["b", "c", "d"]})
    bins = ex.group_by_prediction_bins(grp, model.predict(ds.X_test), n_bins=5)
    paths = [
        pl.waterfall(sv[0], "t", out.figure("w.png")),
        pl.beeswarm(sv, out.figure("b.png")),
        pl.dependence(sv, "b", "c", out.figure("d.png")),
        pl.group_stack_bins(bins, ["G1", "G2"], base, out.figure("g.png")),
    ]
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)
