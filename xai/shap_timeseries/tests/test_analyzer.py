"""ShapAnalyzer(Facade)単体のテスト。合成データで、このプロジェクトのデータに依存しないことを確かめる。"""
import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor

from src.xai_platform import Offset, ShapAnalyzer, Unit

TOL = 1e-6


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.uniform(0, 10, n),
                      "c": rng.integers(0, 3, n).astype(float), "level": rng.uniform(50, 150, n)})
    y = 3 * X["a"] + np.sin(X["b"]) * X["c"] + rng.normal(scale=0.3, size=n)
    model = LGBMRegressor(n_estimators=80, num_leaves=7, verbose=-1, random_state=0)
    model.fit(X.iloc[:300], y.iloc[:300])
    return X, y, model


@pytest.mark.parametrize("with_offset", [False, True])
def test_additivity_and_groups(data, with_offset):
    X, y, model = data
    offset = Offset("水準", lambda Z: Z["level"], features=("level",)) if with_offset else None
    az = ShapAnalyzer(model, X.iloc[:300], unit=Unit("単位", digits=2),
                      groups={"主": ["a"], "副": ["b"]}, offset=offset)
    # 指定しなかった c・level は「その他」、補正項は同名グループに入る
    assert az.groups["その他"] == ["c", "level"]
    assert ("水準" in az.groups) == with_offset
    check = az.explain(X.iloc[300:], actual=y.iloc[300:])
    assert check["合格"].all()
    assert np.allclose(az.base + az.contributions().sum(axis=1), az.pred, atol=TOL)
    expected = model.predict(X.iloc[300:]) + (X["level"].iloc[300:] if with_offset else 0)
    assert np.allclose(az.pred, expected, atol=TOL)


def test_compare_identity(data):
    X, y, model = data
    az = ShapAnalyzer(model, X.iloc[:300])
    az.explain(X.iloc[300:], actual=y.iloc[300:])
    idx = X.index[300:]
    tbl = az.compare_table(zip(idx[1:], idx[:-1]))
    assert tbl["恒等式の誤差"].max() < TOL


def test_whatif_and_outputs(data, tmp_path):
    X, y, model = data
    az = ShapAnalyzer(model, X.iloc[:300], out_dir=tmp_path,
                      offset=Offset("水準", lambda Z: Z["level"], features=("level",)))
    az.explain(X.iloc[300:], actual=y.iloc[300:])
    key = X.index[310]
    scen = {"S1 a+1": {"a": ("abs", 1.0)}, "S2 a+1×b+2": {"a": ("abs", 1.0), "b": ("abs", 2.0)}}
    r = az.whatif(key, scen, name="t")
    assert np.allclose(r.tables["シナリオ別の結果"]["Δ予測値(同時)"],
                       r.tables["ΔSHAP"].sum(axis=1), atol=TOL)
    with pytest.raises(ValueError):
        az.whatif(key, {"NG": {"level": ("abs", 1.0)}})
    az.whatif_period(scen, name="period")
    az.decompose(key, name="k")
    az.compare(key, X.index[309], name="k")
    az.overview(dependence=["a"], name="all")
    for f in ["whatif_multi_t.png", "period.png", "waterfall_k.png", "group_waterfall_k.png",
              "delta_k.png", "beeswarm_all.png", "dependence_a.png"]:
        assert (tmp_path / "figures" / f).exists(), f
    assert (tmp_path / "tables" / "decomposition_k.csv").exists()


def test_requires_explain(data):
    X, _, model = data
    with pytest.raises(RuntimeError):
        ShapAnalyzer(model, X).decompose(X.index[0])


def test_run_datetime_index(data, tmp_path):
    """おまかせ実行: 日時 index なら推移・前行比も含めて一通り出力する。"""
    X, y, model = data
    X = X.set_index(pd.date_range("2020-01-01", periods=len(X), freq="D"))
    y = y.set_axis(X.index)
    model2 = LGBMRegressor(n_estimators=50, num_leaves=7, verbose=-1, random_state=0)
    model2.fit(X.iloc[:300], y.iloc[:300])
    from src.xai_platform import quick_analyze
    az, res = quick_analyze(model2, X.iloc[:300], X.iloc[300:], y.iloc[300:], out_dir=tmp_path,
                            whatif_features=["a", "b"], freq="M")
    assert res.tables["加法性チェック"]["合格"].all()
    assert len(res.tables["代表行"]) == 4  # 最大・最小・中央値付近・誤差最大
    figs = {p.name for p in (tmp_path / "figures").iterdir()}
    for f in ["waterfall_auto_1.png", "group_waterfall_auto_4.png", "beeswarm_auto.png",
              "timeline_auto.png", "delta_auto_up.png", "delta_auto_down.png",
              "whatif_single_auto_1.png", "whatif_multi_auto_1.png", "whatif_period_auto.png"]:
        assert f in figs, f
    assert (tmp_path / "manifest.md").exists()
    # 続けて個別メソッドも使える
    assert az.compare_table([(X.index[350], X.index[349])])["恒等式の誤差"].max() < TOL


def test_run_without_saving(data):
    """out_dir なし・日時でない index・実測なしでも動き、ファイルは作らない。"""
    X, _, model = data
    az = ShapAnalyzer(model, X.iloc[:300])
    res = az.run(X.iloc[300:], keys=[X.index[305], X.index[320]], n_features=2)
    assert list(res.tables["代表行"]["キー"]) == [X.index[305], X.index[320]]
    assert res.figures == []
    assert "全行: シナリオ別の総量" in res.tables
