"""分析ステップ。ノートブックからは各ステップを呼ぶだけにする。

通し実行: python -m src.pipeline
各ステップは前提(モデル学習・SHAP計算)が無ければ自動で用意するので、単独でも呼べる。
SHAP の計算・分解・What-if・図は xai_platform.ShapAnalyzer(Facade)に任せ、
ここではこのデータ固有の設定(代表日・比較時点・シナリオ・ファイル名)を渡す。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import decompose as dc
from . import plots
from .config import OUTPUTS, load_config, unit
from .data import load_daily, time_split
from .explain import date_label, make_analyzer, ridge_check, ridge_explanation, select_days
from .features import build_features
from .model import accuracy_table, fit_tuned, naive_predictions
from .whatif import parse_scenarios
from .xai_platform import (APP_SOURCE, MANIFEST, OutputDirs, ShapAnalyzer, StepResult,
                           analysis_step, save_csv, save_json)

LABELS = {"lgbm": "LightGBM", "ridge": "Ridge"}
WHATIF_SOURCE = "xai_platform.analyzer.ShapAnalyzer.whatif"


class BikeAnalysis:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        np.random.seed(self.cfg["seed"])
        self.unit = unit(self.cfg)
        self.out = OutputDirs(OUTPUTS).ensure()
        self.df = load_daily(self.cfg)
        self.fs = build_features(self.df, self.cfg)
        split = time_split(self.df.index, self.cfg["split"]["test_ratio"])
        X, y = self.fs.X, self.fs.y
        self.X_train, self.y_train = X[X.index < split.test_start], y[y.index < split.test_start]
        self.X_test, self.y_test = X[X.index >= split.test_start], y[y.index >= split.test_start]
        self.test_start = split.test_start
        self.tf = self.cfg["explain_transform"]
        self.ar_group = self.cfg["groups"]["ar"]
        self.exog_group = self.cfg["groups"]["exog"]
        self.models: dict = {}
        self.key: dict = {}
        self.az: ShapAnalyzer | None = None
        self._days = None

    # ------------------------------------------------------------ 前提の用意
    def _model(self, kind: str, tf: str):
        name = f"{LABELS[kind]}[{tf}]"
        if name not in self.models:
            self.models[name], _ = fit_tuned(kind, tf, self.X_train, self.y_train, self.cfg)
        return self.models[name]

    def _explain(self) -> ShapAnalyzer:
        """説明用モデルの ShapAnalyzer を作り、全期間の SHAP を計算する(加法性チェック込み)。"""
        if self.az is None:
            self.model = self._model("lgbm", self.tf)
            self.az = make_analyzer(self.model, self.X_train, self.cfg, self.fs.groups,
                                    self.unit, OUTPUTS)
            self.az.explain(self.fs.X, actual=self.fs.y)
        return self.az

    def days(self) -> pd.DataFrame:
        az = self._explain()
        if self._days is None:
            self._days = select_days(az.pred, self.fs.y, self.X_test.index,
                                     self.cfg["representatives"])
        return self._days

    # ------------------------------------------------------------ 1. データ確認
    def step_data(self) -> StepResult:
        """データ確認: 期間・分割・特徴量の説明表・系列の推移。"""
        res = StepResult()
        with analysis_step("1. データ確認"):
            y = self.df[self.cfg["data"]["target"]]
            rows = []
            for name, s in [("全期間", y), ("訓練(特徴量あり)", self.y_train), ("テスト", self.y_test)]:
                rows.append({"区分": name, "開始": s.index[0].date(), "終了": s.index[-1].date(),
                             "日数": len(s), "平均": s.mean(), "最小": s.min(), "最大": s.max()})
            info = pd.DataFrame(rows).set_index("区分")
            save_csv(info, self.out.table("data_periods.csv"), source=APP_SOURCE)
            desc = ShapAnalyzer.feature_table(self.cfg["descriptions"], self.fs.groups)
            save_csv(desc, self.out.table("feature_descriptions.csv"), index=False)
            res.tables.update({"期間と分割": info, "特徴量の説明": desc})
            res.figures.append(plots.series_with_split(
                y, self.test_start, self.out.figure("series_split.png"),
                "日次レンタル数の推移と訓練/テスト期間", self.unit))
        return res

    # ------------------------------------------------------------ 2. 精度
    def step_baseline(self) -> StepResult:
        """目的変数の変換 × モデル(LightGBM / Ridge)を TimeSeriesSplit で調整し、Naive と比較する。"""
        res = StepResult()
        with analysis_step("2. 精度比較(Naive含む)"):
            preds = naive_predictions(self.X_test, self.cfg["naive"])
            cv, params = {}, {}
            for kind in LABELS:
                for tf in self.cfg["target_transforms"]:
                    name = f"{LABELS[kind]}[{tf}]"
                    m, cv[name] = fit_tuned(kind, tf, self.X_train, self.y_train, self.cfg)
                    self.models[name], params[name] = m, m.params
                    preds[name] = m.predict(self.X_test)
            acc = accuracy_table(self.y_test, preds, cv, params)
            save_csv(acc, self.out.table("accuracy.csv"))
            drop = self.y_test.idxmin()
            keep = self.y_test.index != drop
            acc_ex = accuracy_table(self.y_test[keep], {k: v[keep] for k, v in preds.items()})
            save_csv(acc_ex, self.out.table("accuracy_excl_outlier.csv"))
            res.tables.update({"テスト期間の精度": acc,
                               f"参考: {drop.date()}(実測{self.y_test[drop]:.0f})を除外": acc_ex})
            main = f"LightGBM[{self.tf}]"
            show = list(self.cfg["naive"]) + [main, f"Ridge[{self.tf}]"]
            res.figures.append(plots.test_predictions(
                self.y_test, {k: pd.Series(preds[k], self.y_test.index) for k in show},
                self.out.figure("test_predictions.png"), "テスト期間の実測値と予測値", self.unit))
            naive_best = acc.loc[list(self.cfg["naive"]), "RMSE"].min()
            self.key["精度"] = {"説明に使うモデル": main, "RMSE": acc.loc[main, "RMSE"],
                               "MAE": acc.loc[main, "MAE"], "sMAPE(%)": acc.loc[main, "sMAPE(%)"],
                               "Naive最良RMSE": naive_best,
                               "Naiveに勝つ": bool(acc.loc[main, "RMSE"] < naive_best),
                               "除外参考日": str(drop.date())}
        return res

    # ------------------------------------------------------------ 3. SHAP と検証
    def step_explain(self) -> StepResult:
        """SHAP 計算、加法性チェック(モデル出力/予測値)、Ridge での係数×(x−平均) 一致検証。"""
        az = self._explain()
        res = StepResult()
        with analysis_step("3. SHAP計算と検証"):
            add = az.additivity(self.fs.X)
            save_csv(add, self.out.table("additivity_check.csv"))
            ridge = self._model("ridge", self.tf)
            rc = ridge_check(ridge, self.X_train, self.X_test)
            save_csv(rc, self.out.table("ridge_shap_check.csv"))
            # LightGBM と Ridge のグループ別平均|寄与|(テスト期間)
            sv_r, base_r = ridge_explanation(ridge, self.X_train, self.X_test, az.offset_ref,
                                             self.cfg["level_name"])
            g_r = ShapAnalyzer.group_sum(sv_r, self.X_test.index, az.groups)
            g_l = az.contributions().loc[self.X_test.index]
            comp = pd.DataFrame({"LightGBM 平均|寄与|": g_l.abs().mean(),
                                 "Ridge 平均|寄与|": g_r.abs().mean(),
                                 "LightGBM 平均寄与": g_l.mean(), "Ridge 平均寄与": g_r.mean()})
            comp.loc["ベースライン"] = [np.nan, np.nan, az.base, base_r]
            comp.index.name = "グループ"
            save_csv(comp, self.out.table("group_lgbm_vs_ridge.csv"), source=APP_SOURCE)
            res.tables.update({"加法性チェック": add, "Ridge: SHAP と 係数×(x−平均) の一致": rc,
                               "グループ別寄与の比較(テスト期間)": comp})
            train_mean = float(self.model.predict(self.X_train).mean())
            res.texts.append(f"ベースライン(元スケール)= E[f] {az.base_raw:,.1f} + "
                             f"訓練期間の roll7_mean 平均 {az.offset_ref:,.1f} = "
                             f"{az.base:,.1f} 件/日(訓練期間の平均予測 {train_mean:,.1f})")
            self.key["検証"] = {
                "ベースライン": az.base, "E[f]": az.base_raw, "訓練平均roll7": az.offset_ref,
                "訓練期間の平均予測": train_mean,
                "加法性_最大誤差": float(add["最大絶対誤差"].max()),
                "加法性_合格": bool(add["合格"].all()),
                "Ridge一致_最大差": float(rc.iloc[:, -1].max()),
                "Ridgeベースライン誤差": float(rc.attrs["ベースライン誤差"])}
        return res

    # ------------------------------------------------------------ 4. ミクロ
    def step_micro(self) -> StepResult:
        """代表日の分解: 分解表・個別特徴量のウォーターフォール・グループ単位の図・日本語要約。"""
        days = self.days()
        az = self._explain()
        res = StepResult()
        with analysis_step("4. 代表日の分解(ミクロ)"):
            save_csv(days, self.out.table("representative_days.csv"), index=False)
            res.tables["代表日と選定基準"] = days
            sentences = {}
            for k, r in days.iterrows():
                d = r["日付"]
                one = az.decompose(d, label=f"{r['区分']} {date_label(d)}", name=f"day{k + 1}")
                sentences[r["区分"]] = one.texts[0]
                res.tables[f"{r['区分']} {date_label(d)}"] = one.tables["分解表"]
                res.texts += one.texts
                res.figures += one.figures
            save_json(sentences, self.out.table("representative_sentences.json"))
            self.key["代表日の要約"] = sentences
        return res

    # ------------------------------------------------------------ 5. マクロ
    def step_macro(self) -> StepResult:
        """時間方向の構成: 積み上げ時系列・季節/月/曜日の平均寄与・STL比較・beeswarm・依存プロット。"""
        az = self._explain()
        res = StepResult()
        u = self.unit
        with analysis_step("5. 時間方向の構成(マクロ)"):
            save_csv(az.table, self.out.table("contribution_timeseries.csv"))
            start = pd.Timestamp(self.cfg.get("macro_start", "2012-01-01"))
            for kw in [dict(start=start, name="stacked_contributions",
                            title=f"予測値の構成の推移: ベースライン + グループ別寄与"
                                  f"({start:%Y-%m}〜、訓練期間は学習内)"),
                       dict(start=self.test_start, name="stacked_contributions_test",
                            title="テスト期間の予測値の構成: ベースライン + グループ別寄与")]:
                res.figures += az.timeline(marker=self.test_start, marker_label="テスト期間→",
                                           **kw).figures
            for by in ("season", "month", "weekday"):
                keys, key_name = dc.calendar_keys(self.fs.X.index, by)
                one = az.profile(keys, key_name=key_name, name=by, heatmap=by != "season",
                                 title=f"{key_name}別のグループ平均寄与(全期間、{u.label})")
                res.tables.update({f"{k}(全期間)": v for k, v in one.tables.items()})
                res.figures += one.figures
            stl = dc.stl_components(self.df[self.cfg["data"]["target"]])
            weekly, annual = dc.stl_comparison(stl, az.contributions("feature"), az.contributions(),
                                               self.cfg["calendar_parts"], self.ar_group,
                                               self.exog_group)
            save_csv(weekly, self.out.table("stl_weekly_comparison.csv"))
            save_csv(annual, self.out.table("stl_annual_comparison.csv"))
            res.tables.update({"STL週周期 vs カレンダー寄与": weekly, "STL年周期 vs モデル寄与": annual})
            res.figures.append(plots.stl_comparison(weekly, annual,
                                                    self.out.figure("stl_comparison.png"), u))
            self.key["STL相関"] = {
                "週周期 vs カレンダー(週)": float(weekly.corr().iloc[0, 1]),
                **{f"年周期 vs {c}": float(annual.corr().iloc[0, k])
                   for k, c in enumerate(annual.columns) if k}}
            # 補助: beeswarm・平均|SHAP|(テスト期間)と、外生変数の依存プロット(全期間)
            ov = az.overview(rows=self.X_test.index, dependence=self.fs.exog, label="テスト期間",
                             name="test")
            res.tables.update(ov.tables)
            res.texts += ov.texts
            res.figures += ov.figures
            self.key["依存プロットの色"] = az.partners
        return res

    # ------------------------------------------------------------ 6. Δ分解
    def step_delta(self) -> StepResult:
        """前日比・前週同曜日比・前年同期比の変化を ΔSHAP(グループ別)で分解する。"""
        days = self.days()
        az = self._explain()
        res = StepResult()
        g = az.group_names
        with analysis_step("6. 変化の要因分解(ΔSHAP)"):
            means, deltas = {}, {}
            for label, lag in self.cfg["delta_lags"].items():
                d = dc.delta_decomposition(az, lag, self.X_test.index)
                deltas[label] = d
                save_csv(d, self.out.table(f"delta_lag{lag}.csv"))
                means[label] = d[["Δ予測値"] + [f"Δ{x}" for x in g] + ["Δ実測値", "恒等式の誤差"]].agg(
                    lambda s: s.abs().max() if s.name == "恒等式の誤差" else s.mean())
            mt = pd.DataFrame(means).T
            mt.index.name = "比較(テスト期間の平均。恒等式の誤差は最大値)"
            save_csv(mt, self.out.table("delta_summary.csv"), source=APP_SOURCE)
            res.tables["比較時点ごとの平均Δ(テスト期間)"] = mt
            self.key["Δ恒等式_最大誤差"] = float(mt["恒等式の誤差"].max())

            def fig(lag_label, t, tag):
                d = deltas[lag_label].loc[t]
                s = d["比較日"]
                title = (f"{lag_label}の要因分解: {date_label(t)}(Δ予測値 "
                         f"{self.unit.fmt(d['Δ予測値'], True)}、Δ実測 {self.unit.fmt(d['Δ実測値'], True)})")
                return az.compare(t, s, label_t=date_label(t), label_s=date_label(s), title=title,
                                  name=tag).figures

            first = list(self.cfg["delta_lags"])
            d1 = deltas[first[0]]
            res.figures += fig(first[0], d1["Δ予測値"].idxmax(), "lag1_up")
            res.figures += fig(first[0], d1["Δ予測値"].idxmin(), "lag1_down")
            top_day = days.iloc[0]["日付"]
            for label in first[1:]:
                if top_day in deltas[label].index:
                    res.figures += fig(label, top_day, f"lag{self.cfg['delta_lags'][label]}_top")
            top, summ = dc.top_change_drivers(d1, g, self.cfg["delta_top_n"])
            save_csv(top, self.out.table("delta_lag1_top_days.csv"))
            save_csv(summ, self.out.table("delta_lag1_drivers.csv"))
            res.tables.update({f"前日比の上昇・下落 上位{self.cfg['delta_top_n']}日": top,
                               "主因グループの集計": summ})
            self.key["前日比の主因"] = summ.filter(like="主因の件数").to_dict(orient="index")
        return res

    # ------------------------------------------------------------ 7. What-if
    def step_whatif(self) -> StepResult:
        """外生変数の単一・複数同時 What-if(再予測と ΔSHAP 内訳・相互作用分)と期間全体への適用。"""
        days = self.days()
        az = self._explain()
        res = StepResult()
        wc = self.cfg["whatif"]
        pairs = [tuple(p) for p in wc["correlated_pairs"]]
        singles = parse_scenarios(wc["single"], self.fs.exog)
        multis = parse_scenarios(wc["multi"], self.fs.exog)
        periods = parse_scenarios(wc["period"], self.fs.exog)
        with analysis_step("7. What-if"):
            save_csv(az.whatif_bounds(self.fs.exog), self.out.table("whatif_bounds.csv"))
            single_rows, multi_rows, dshap_rows = [], [], []
            for k, r in days.iterrows():
                d, tag = r["日付"], f"day{k + 1}"
                head = f"{r['区分']} {date_label(d)}"
                one = az.whatif_single(d, singles, label=head, name=tag, correlated_pairs=pairs)
                single_rows.append(one.tables["単独変更の結果"].assign(区分=r["区分"], 日付=d.date()))
                many = az.whatif(d, multis, label=head, name=tag, correlated_pairs=pairs)
                multi_rows.append(many.tables["シナリオ別の結果"].assign(区分=r["区分"], 日付=d.date()))
                dshap_rows.append(many.tables["ΔSHAP"].assign(区分=r["区分"], 日付=d.date()))
                res.figures += one.figures + many.figures
            single = pd.concat(single_rows, ignore_index=True)
            multi = pd.concat(multi_rows, ignore_index=True)
            save_csv(single, self.out.table("whatif_single.csv"), index=False,
                     source="xai_platform.analyzer.ShapAnalyzer.whatif_single")
            save_csv(multi, self.out.table("whatif_multi_summary.csv"), index=False,
                     source=WHATIF_SOURCE)
            save_csv(pd.concat(dshap_rows), self.out.table("whatif_multi_dshap.csv"),
                     source=WHATIF_SOURCE)
            res.tables.update({"単独変更のΔ予測値(代表日)": single.pivot_table(
                index=["区分", "日付"], columns="シナリオ", values="Δ予測値", sort=False),
                "同時変更のΔ予測値と相互作用分(代表日)": multi.pivot_table(
                    index=["区分", "日付"], columns="シナリオ",
                    values=["Δ予測値(同時)", "相互作用分(同時−単独合計)"], sort=False)})
            # 期間全体(テスト期間の全日)
            per = az.whatif_period(periods, rows=self.X_test.index, freq="M", name="whatif_period",
                                   title="テスト期間の全日に外生変数シナリオを適用したときのΔ予測値")
            psum = per.tables["シナリオ別の総量"]
            monthly = per.tables["期間別の平均Δ予測値"].rename_axis("月")
            save_csv(psum, self.out.table("whatif_period_summary.csv"))
            save_csv(per.tables["行別のΔ予測値"], self.out.table("whatif_period_daily.csv"))
            save_csv(monthly, self.out.table("whatif_period_monthly.csv"))
            save_csv(per.tables["特徴量別の平均ΔSHAP"], self.out.table("whatif_period_mean_dshap.csv"))
            res.tables.update({"期間全体への適用(テスト期間)": psum, "月別の1日平均Δ": monthly})
            res.figures += per.figures
            self.key["What-if"] = {"期間全体": psum.to_dict(orient="index"),
                                   "月別": monthly.to_dict(orient="index")}
        return res

    def finalize(self) -> None:
        save_json(self.key, self.out.table("key_results.json"))
        MANIFEST.save(OUTPUTS)

    def all_steps(self):
        return [self.step_data, self.step_baseline, self.step_explain, self.step_micro,
                self.step_macro, self.step_delta, self.step_whatif]


def main() -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    a = BikeAnalysis()
    for step in a.all_steps():
        print(f"\n================ {step.__doc__.splitlines()[0]}")
        r = step()
        for t in r.texts:
            print(t)
        for name, df in r.tables.items():
            print(f"\n■ {name}")
            print(df.to_string(max_colwidth=90))
    a.finalize()


if __name__ == "__main__":
    main()
