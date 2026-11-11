"""California Housing の分析パイプライン。

各ステップは xai_platform の I/F を呼び出し、outputs/tables(CSV)と outputs/figures(PNG)に
保存したうえで StepResult(表・図・テキスト)を返す。ノートブックでは result.show() で表示する。

コマンドラインからの通し実行:
    python -m src.app_california.pipeline
"""
from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from src.xai_platform import explain as ex
from src.xai_platform import plots as pl
from src.xai_platform import whatif as wi
from src.xai_platform.data import description_table
from src.xai_platform.io import StepResult, save_csv, save_json
from src.xai_platform.model import evaluation_table
from src.xai_platform.provenance import MANIFEST, interface_name

from . import config as C
from .data import load_california
from .model import train_lgbm, train_ridge


class CaliforniaAnalysis:
    def __init__(self):
        self.out = C.OUTPUTS.ensure()
        self.unit = C.UNIT
        self.key: dict = {}  # レポート用の主要数値(key_results.json)

    # ------------------------------------------------------------ 1. データ
    def step_data(self) -> StepResult:
        self.ds = load_california()
        desc = description_table(C.FEATURE_DESCRIPTIONS)
        save_csv(desc, self.out.table("feature_descriptions.csv"), index=False)
        cap_ratio = round(float((self.ds.y_train >= C.TARGET_CAP).mean()), 4)
        return StepResult(tables={"特徴量の説明": desc},
                          texts=[f"train {self.ds.X_train.shape} test {self.ds.X_test.shape}",
                                 f"目的変数が上限{C.TARGET_CAP}で打ち切られている割合(train): {cap_ratio}"])

    # ------------------------------------------------------------ 2. モデル
    def step_models(self) -> StepResult:
        self.lgbm = train_lgbm(self.ds)
        self.ridge = train_ridge(self.ds)
        perf = evaluation_table({"LightGBM": self.lgbm, "Ridge": self.ridge}, self.ds)
        save_csv(perf, self.out.table("model_performance.csv"))
        self.key["performance"] = perf.round(4).to_dict(orient="index")
        return StepResult(tables={"テストデータでの性能": perf})

    # ------------------------------------------------------------ 3. SHAP と検証
    def step_shap_validation(self) -> StepResult:
        ds = self.ds
        self.explainer = ex.make_tree_explainer(self.lgbm)
        self.base = ex.baseline(self.explainer)
        train_mean_pred = float(self.lgbm.predict(ds.X_train).mean())
        self.sv = ex.compute_shap(self.explainer, ds.X_test)
        self.pred_test = pd.Series(self.lgbm.predict(ds.X_test), index=ds.X_test.index)
        self.pos = {idx: i for i, idx in enumerate(ds.X_test.index)}

        check = ex.additivity_check(self.explainer, self.lgbm, ds.X_test, self.sv)
        add_sum = ex.additivity_summary(check)
        save_csv(check, self.out.table("additivity_check_test.csv"))
        ridge_chk = ex.linear_shap_check(self.ridge, ds.X_train, ds.X_test)
        save_csv(ridge_chk, self.out.table("ridge_shap_vs_coef.csv"))

        self.key.update(baseline=self.base, train_mean_pred=train_mean_pred, additivity=add_sum,
                        ridge_max_diff=float(ridge_chk.iloc[:, 1].max()),
                        ridge_base_err=float(ridge_chk.attrs["ベースライン誤差"]))
        return StepResult(
            tables={"Ridge: SHAP値 vs 係数×(x−平均)": ridge_chk},
            texts=[f"ベースライン expected_value = {self.base:.6f} / 訓練データの平均予測 = "
                   f"{train_mean_pred:.6f} / 差 = {abs(self.base - train_mean_pred):.2e}",
                   f"加法性チェック: {add_sum}",
                   f"Ridge: ベースライン誤差 {ridge_chk.attrs['ベースライン誤差']:.2e}"])

    # ------------------------------------------------------------ 4. ミクロ
    def step_micro(self) -> StepResult:
        quantiles = {label: q for label, q in C.REPRESENTATIVES.values()}
        self.reps = ex.select_representatives(self.pred_test, quantiles)
        self.reps["タグ"] = list(C.REPRESENTATIVES)
        save_csv(self.reps.drop(columns="タグ"), self.out.table("representative_samples.csv"),
                 index=False)
        res = StepResult(tables={"代表サンプル": self.reps.drop(columns="タグ")})
        sentences = []
        for _, r in self.reps.iterrows():
            idx, tag = r["サンプルindex"], r["タグ"]
            row = self.sv[self.pos[idx]]
            tbl = ex.decomposition_table(row, self.base, self.unit)
            save_csv(tbl, self.out.table(f"decomposition_{tag}.csv"), index=False)
            res.figures.append(pl.waterfall(row, f"{r['区分']}: test index {idx} の予測値の分解",
                                            self.out.figure(f"waterfall_{tag}.png"), self.unit))
            s = ex.summary_sentence(row, self.base, self.unit)
            y = float(self.ds.y_test.loc[idx])
            sentences.append({"区分": r["区分"], "サンプルindex": idx, "実測値": y, "要約": s})
            res.tables[f"分解表: {r['区分']}(index {idx}、実測値 {y:.3f})"] = tbl
            res.texts.append(f"■ {r['区分']}: {s}")
        save_csv(pd.DataFrame(sentences), self.out.table("decomposition_summaries.csv"), index=False,
                 source=interface_name(ex.summary_sentence))
        self.key["representatives"] = sentences
        return res

    def _rep_x(self, tag: str) -> pd.Series:
        idx = self.reps.set_index("タグ").loc[tag, "サンプルindex"]
        return self.ds.X_test.loc[idx]

    # ------------------------------------------------------------ 5. What-if
    def step_whatif_single(self) -> StepResult:
        lo, hi = C.BOUNDS_Q
        self.sd = wi.robust_sd(self.ds.X_train, lo, hi)
        self.bounds = wi.feature_bounds(self.ds.X_train, lo, hi)
        ranges = wi.ranges_table(self.ds.X_train, lo, hi)
        save_csv(ranges, self.out.table("whatif_sd_bounds.csv"))
        res = StepResult(tables={"頑健σと動かせる範囲": ranges})

        self.key["whatif_single"] = {}
        for _, r in self.reps.iterrows():
            tag = r["タグ"]
            one = wi.single_variable_whatif(self.lgbm, self._rep_x(tag), self.sd, self.bounds,
                                            "sd", 1.0, correlated_pairs=C.CORRELATED_PAIRS)
            save_csv(one, self.out.table(f"whatif_single_sd_{tag}.csv"), index=False)
            res.figures.append(pl.tornado(
                one, f"{r['区分']}: 各特徴量を単独で+1σ動かしたときのΔ予測値\n"
                     f"(元の予測 {one['元の予測'].iloc[0]:.3f})",
                self.out.figure(f"tornado_sd_{tag}.png"), self.unit))
            self.key["whatif_single"][tag] = one[["特徴量", "元の値", "適用値", "Δ予測値", "外挿警告"]
                                                 ].round(4).to_dict(orient="records")
            if tag == C.WHATIF_PCT_TARGET:
                res.tables[f"+1σ: {r['区分']}"] = one

        tag = C.WHATIF_PCT_TARGET
        pct = wi.single_variable_whatif(self.lgbm, self._rep_x(tag), self.sd, self.bounds,
                                        "pct", 0.10, exclude=C.PCT_EXCLUDED,
                                        correlated_pairs=C.CORRELATED_PAIRS)
        save_csv(pct, self.out.table(f"whatif_single_pct10_{tag}.csv"), index=False)
        res.figures.append(pl.tornado(
            pct, f"中央値付近サンプル: 各特徴量を単独で+10%動かしたときのΔ予測値\n"
                 f"(緯度・経度は対象外、元の予測 {pct['元の予測'].iloc[0]:.3f})",
            self.out.figure(f"tornado_pct10_{tag}.png"), self.unit))
        self.key[f"whatif_pct10_{tag}"] = pct[["特徴量", "元の値", "適用値", "Δ予測値"]
                                              ].round(4).to_dict(orient="records")
        return res

    def step_scenarios(self) -> StepResult:
        res = StepResult()
        self.key["scenarios"] = {}
        for _, r in self.reps.iterrows():
            tag = r["タグ"]
            summ, dshap, chg = wi.scenario_analysis(self.lgbm, self.explainer, self._rep_x(tag),
                                                    self.sd, self.bounds, C.SCENARIOS,
                                                    C.CORRELATED_PAIRS)
            assert np.allclose(summ["ΣΔSHAP"], summ["Δ予測値(同時)"], atol=1e-6)
            save_csv(summ, self.out.table(f"whatif_scenarios_{tag}.csv"), index=False)
            save_csv(dshap, self.out.table(f"whatif_scenarios_dshap_{tag}.csv"))
            save_csv(chg, self.out.table(f"whatif_scenarios_changes_{tag}.csv"), index=False)
            fig = pl.scenario_dshap(summ, dshap, f"{r['区分']}(index {r['サンプルindex']}、元の予測 "
                                    f"{summ['元の予測'].iloc[0]:.3f})のシナリオ分析",
                                    self.out.figure(f"scenario_dshap_{tag}.png"), self.unit)
            self.key["scenarios"][tag] = {
                "summary": summ.drop(columns=["単独Δの内訳"]).round(4).to_dict(orient="records"),
                "dshap": dshap.round(4).to_dict(orient="index"),
                "changes": chg.round(4).to_dict(orient="records")}
            if tag == C.WHATIF_PCT_TARGET:
                res.tables.update({"適用した変更": chg, "シナリオ別の結果": summ, "ΔSHAP": dshap})
                res.figures.append(fig)
        return res

    # ------------------------------------------------------------ 6. マクロ
    def step_global(self) -> StepResult:
        figs = [pl.beeswarm(self.sv, self.out.figure("beeswarm.png"), self.unit,
                            title="各特徴量が予測値を押し上げる/押し下げる分布(テストデータ)"),
                pl.mean_abs_bar(self.sv, self.out.figure("mean_abs_shap.png"), self.unit,
                                title="特徴量ごとの寄与の平均的な大きさ(テストデータ)")]
        self.imp = ex.mean_abs_shap(self.sv)
        save_csv(self.imp.rename("平均|SHAP|").to_frame(), self.out.table("mean_abs_shap.csv"))
        self.key["mean_abs_shap"] = self.imp.round(4).to_dict()
        return StepResult(figures=figs)

    def step_dependence(self) -> StepResult:
        res = StepResult()
        self.key["dependence"] = {}
        for f in list(self.imp.index[:C.TOP_DEPENDENCE]):
            partner = ex.strongest_partner(self.sv, f, self.ds.X_test)
            xlim = tuple(self.bounds.loc[f]) if f in C.DEPENDENCE_CLIP_X else None
            res.figures.append(pl.dependence(self.sv, f, partner,
                                             self.out.figure(f"dependence_{f}.png"), self.unit,
                                             xlim=xlim))
            dt = ex.dependence_table(self.sv, f, unit=self.unit)
            save_csv(dt, self.out.table(f"dependence_{f}.csv"), index=False)
            res.tables[f"{f}(色: {partner})"] = dt
            self.key["dependence"][f] = {"partner": partner,
                                         "table": dt.round(4).to_dict(orient="records")}
        return res

    def step_interactions(self) -> StepResult:
        inter, X_sub = ex.interaction_values(self.explainer, self.ds.X_test, n=C.INTERACTION_N,
                                             seed=C.SEED)
        sv_sub = self.sv.values[[self.pos[i] for i in X_sub.index]]
        feats = list(self.ds.X_test.columns)
        strength = ex.interaction_strength(inter, feats)
        save_csv(strength, self.out.table("interaction_strength.csv"), index=False)
        mvi = ex.main_vs_interaction(inter, feats)
        save_csv(mvi, self.out.table("main_vs_interaction.csv"))
        figs = [pl.interaction_heatmap(
            strength, feats, self.out.figure("interaction_heatmap.png"), self.unit,
            title=f"特徴量ペアの相互作用の強さ(テスト{C.INTERACTION_N:,}件サブサンプル)")]
        for _, r in strength.head(C.TOP_INTERACTION_PAIRS).iterrows():
            f1, f2 = r["特徴量1"], r["特徴量2"]
            figs.append(pl.interaction_pair(inter, X_sub, f1, f2,
                                            self.out.figure(f"interaction_{f1}_{f2}.png"),
                                            self.unit))
        self.key["interaction_top"] = strength.head(5).round(4).to_dict(orient="records")
        self.key["main_vs_interaction"] = mvi.round(4).to_dict(orient="index")
        return StepResult(
            tables={"相互作用の強さ(上位8)": strength.head(8), "主効果と相互作用分": mvi},
            figures=figs,
            texts=["相互作用値の行和とSHAP値の最大差: "
                   f"{float(np.abs(inter.sum(axis=2) - sv_sub).max()):.2e}"])

    def step_groups(self) -> StepResult:
        grp = ex.group_contributions(self.sv, C.FEATURE_GROUPS)
        groups = list(C.FEATURE_GROUPS)
        bins = ex.group_by_prediction_bins(grp, self.pred_test.values, n_bins=C.N_PRED_BINS)
        save_csv(bins, self.out.table("group_contrib_by_pred_decile.csv"))
        figs = [
            pl.group_stack_bins(bins, groups, self.base, self.out.figure("group_stack_deciles.png"),
                                self.unit, C.GROUP_COLORS,
                                title="予測値の構成: ベースライン + 特徴量グループ別の平均寄与(テストデータ)"),
            pl.group_stack_sorted(grp, self.pred_test.values, self.base,
                                  self.out.figure("group_stack_sorted.png"), self.unit,
                                  C.GROUP_COLORS,
                                  title="全テストサンプルの予測値の構成(特徴量グループ別の寄与)",
                                  xlabel="テストサンプル(予測値の低い順に並べたもの)")]
        self.key["group_bins"] = bins.round(4).to_dict(orient="index")
        self.key["group_mean_abs"] = grp.abs().mean().round(4).to_dict()
        return StepResult(tables={"予測値の十分位ビン別の平均寄与": bins}, figures=figs)

    # ------------------------------------------------------------ まとめ
    def save_key_results(self) -> StepResult:
        path = save_json(self.key, self.out.table("key_results.json"),
                         source="アプリ側で作成(各ステップの主要数値を集約)")
        return StepResult(texts=[f"saved {path}"])

    def save_manifest(self) -> StepResult:
        """出力ファイルと生成I/Fの対応表(outputs/manifest.csv, manifest.md)を書き出す。"""
        csv, md = MANIFEST.save(self.out.root)
        return StepResult(tables={"出力ファイルと生成I/F": MANIFEST.to_frame(self.out.root)},
                          texts=[f"saved {csv}", f"saved {md}"])

    STEPS = ("step_data", "step_models", "step_shap_validation", "step_micro",
             "step_whatif_single", "step_scenarios", "step_global", "step_dependence",
             "step_interactions", "step_groups", "save_key_results", "save_manifest")

    def run_all(self, verbose: bool = True) -> None:
        for name in self.STEPS:
            res = getattr(self, name)()
            if verbose:
                print(f"[{name}] 図{len(res.figures)}枚 / 表{len(res.tables)}件")
                for t in res.texts:
                    print("   ", t)


def _as_manifest_step(method):
    """ステップ実行中に保存されたファイルへ「分析ステップ」名を記録させる。"""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with MANIFEST.step(f"{type(self).__name__}.{method.__name__}"):
            return method(self, *args, **kwargs)
    return wrapper


for _name in CaliforniaAnalysis.STEPS:
    setattr(CaliforniaAnalysis, _name, _as_manifest_step(getattr(CaliforniaAnalysis, _name)))


if __name__ == "__main__":
    CaliforniaAnalysis().run_all()
