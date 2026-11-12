"""ShapAnalyzer: xai_platform の入口(Facade)。

木モデルの予測値を SHAP で「ベースライン + 寄与」に分解する一連の処理を、1つのクラスにまとめる。
利用者はこのクラスのメソッドだけを呼べばよく、内部の _core/* を直接使う必要はない。

    az = ShapAnalyzer(model, X_train, unit=Unit("件/日", digits=0), groups=..., out_dir="outputs")
    az.explain(X, actual=y)          # SHAP 計算 + 加法性チェック(以降のメソッドの前提)
    az.decompose(key)                # 1行の分解表 + 要約文 + ウォーターフォール図
    az.compare(t, s)                 # 2行の差の分解(Δ予測値 = ΣΔ寄与)
    az.timeline() / az.profile(...)  # 行の並びに沿った寄与の推移 / 区分ごとの平均寄与
    az.overview(dependence=[...])    # beeswarm・平均|SHAP|・依存プロット
    az.whatif_single / whatif / whatif_period   # 特徴量を動かしたときの再予測と ΔSHAP
    az.save_manifest()               # 出力ファイルと生成 I/F の対応表

予測値の定義:
    予測値 = model.predict(X) + offset(X)
offset はモデルの外で足す補正項(例: 目的変数を「直近平均からの差」にしたときの直近平均)。
offset を「訓練データでの平均」と「そこからの差」に分け、前者をベースラインに、後者を
offset.name という擬似特徴量の寄与として扱うので、
    ベースライン + Σ寄与 = 予測値
が元の単位のまま成り立つ。offset が無ければ通常の TreeExplainer の分解と同じ。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Hashable, Iterable

import numpy as np
import pandas as pd
import shap

from ._core import decompose as dc
from ._core import plots
from ._core.explain import (DEFAULT_ADDITIVITY_TOL, baseline, dependence_table,
                            description_table, linear_shap_check, make_tree_explainer,
                            mean_abs_shap, strongest_partner)
from ._core.io import OutputDirs, StepResult, save_csv
from ._core.provenance import MANIFEST, produces_table
from ._core.units import Unit
from ._core.whatif import (Scenario, feature_bounds, period_analysis, robust_sd,
                           scenario_analysis)

OTHER_GROUP = "その他"


def key_label(k: Hashable) -> str:
    """行のキーの表示名(時刻が 0 時ちょうどの日時は日付だけにする)。"""
    if isinstance(k, pd.Timestamp) and k == k.normalize():
        return f"{k:%Y-%m-%d}"
    return str(k)


@dataclass(frozen=True)
class Offset:
    """モデルの外で予測値に足す補正項。予測値 = model.predict(X) + func(X)。

    name:     分解表・図に出す擬似特徴量の名前
    func:     X → 補正値(行ごと)
    features: func が参照する列。What-if でこれらを動かすと ΣΔSHAP と Δ予測値がずれるので禁止する
    group:    擬似特徴量を入れるグループ(None なら name と同名のグループを作る)
    """
    name: str
    func: Callable[[pd.DataFrame], np.ndarray | pd.Series]
    features: tuple[str, ...] = ()
    group: str | None = None


class _Predictor:
    """scenario_analysis が呼ぶ predict を「最終の予測値」にするための薄い入れ物。"""

    def __init__(self, predict):
        self.predict = predict


class ShapAnalyzer:
    """木モデル + (任意の)補正項の予測値を SHAP で分解する Facade。"""

    def __init__(self, model, X_train: pd.DataFrame, *, unit: Unit | None = None,
                 groups: dict[str, list[str]] | None = None, offset: Offset | None = None,
                 out_dir: str | Path | None = None, group_colors: dict[str, str] | None = None,
                 bounds_quantiles: tuple[float, float] = (0.01, 0.99),
                 tol: float = DEFAULT_ADDITIVITY_TOL):
        """
        model:            学習済みの木モデル(LightGBM / XGBoost / sklearn の木など)
        X_train:          訓練データの特徴量(ベースライン・What-if の範囲の基準)
        unit:             目的変数の単位と表示桁数
        groups:           {グループ名: [特徴量]}。漏れた特徴量は「その他」にまとめる。None なら特徴量ごと
        offset:           モデルの外で足す補正項(Offset)。None なら補正なし
        out_dir:          図表の保存先(figures/ と tables/ を作る)。None なら保存しない
        group_colors:     積み上げ図でのグループの色
        bounds_quantiles: What-if で動かしてよい範囲(訓練データの分位点)
        tol:              加法性チェックの許容誤差
        """
        self.model, self.X_train = model, X_train
        self.unit = unit or Unit()
        self.offset = offset
        self.colors = group_colors
        self.tol = tol
        self.tree = make_tree_explainer(model)
        self.offset_ref = float(np.mean(self._offset_values(X_train))) if offset else 0.0
        self.groups = self._resolve_groups(groups)
        self.out = OutputDirs(Path(out_dir)).ensure() if out_dir is not None else None
        lo, hi = bounds_quantiles
        self.sd, self.bounds = robust_sd(X_train, lo, hi), feature_bounds(X_train, lo, hi)
        self.X: pd.DataFrame | None = None

    # ============================================================ 基本
    def _resolve_groups(self, groups):
        feats = list(self.X_train.columns)
        if groups is None:
            out = {f: [f] for f in feats}
        else:
            out = {g: list(fs) for g, fs in groups.items()}
            listed = [f for fs in out.values() for f in fs]
            unknown = sorted(set(listed) - set(feats))
            if unknown:
                raise ValueError(f"groups に存在しない特徴量がある: {unknown}")
            rest = [f for f in feats if f not in listed]
            if rest:
                out.setdefault(OTHER_GROUP, []).extend(rest)
        if self.offset:
            out.setdefault(self.offset.group or self.offset.name, []).append(self.offset.name)
        return out

    def _offset_values(self, X: pd.DataFrame) -> np.ndarray:
        if self.offset is None:
            return np.zeros(len(X))
        return np.asarray(self.offset.func(X), dtype=float)

    @property
    def group_names(self) -> list[str]:
        return list(self.groups)

    @property
    def base_raw(self) -> float:
        """モデル出力のベースライン E[f](訓練データでのモデル出力の平均)。"""
        return baseline(self.tree)

    @property
    def base(self) -> float:
        """予測値のベースライン = E[f] + 訓練データでの補正項の平均。"""
        return self.base_raw + self.offset_ref

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        """モデル出力(補正項を足す前)。"""
        return self.model.predict(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """予測値 = モデル出力 + 補正項。"""
        return self.predict_raw(X) + self._offset_values(X)

    def shap_raw(self, X: pd.DataFrame) -> shap.Explanation:
        """モデル出力の SHAP(補正項を含まない)。"""
        return self.tree(X)

    def shap(self, X: pd.DataFrame) -> shap.Explanation:
        """予測値の SHAP。補正項があれば最終列に擬似特徴量(offset.name)を加える。"""
        sv = self.shap_raw(X)
        values, data, names = sv.values, X.to_numpy(float), list(X.columns)
        if self.offset:
            off = self._offset_values(X)
            values = np.hstack([values, (off - self.offset_ref)[:, None]])
            data = np.hstack([data, off[:, None]])
            names.append(self.offset.name)
        return shap.Explanation(values=values, base_values=np.full(len(X), self.base),
                                data=data, feature_names=names)

    # ============================================================ 計算と検証
    def explain(self, X: pd.DataFrame, actual: pd.Series | None = None) -> pd.DataFrame:
        """X の SHAP を計算して保持し、加法性チェックの表を返す(以降のメソッドの前提)。

        actual(実測値)を渡すと、分解表・推移の図・差の分解に実測値も載る。
        """
        self.X = X
        self.sv = self.shap(X)
        self.pred = pd.Series(self.predict(X), index=X.index)
        self.actual = actual
        self.contrib = dc.group_frame(self.sv, X.index, self.groups)
        self.table = dc.contribution_table(self.contrib, self.base, self.pred, actual, self.tol)
        check = self.additivity(X)
        if not check["合格"].all():
            raise AssertionError(f"加法性チェック不合格:\n{check}")
        return check

    @produces_table
    def additivity(self, X: pd.DataFrame) -> pd.DataFrame:
        """加法性チェック(モデル出力・予測値)の最大絶対誤差と合否。"""
        err_raw = np.abs(self.base_raw + self.shap_raw(X).values.sum(axis=1) - self.predict_raw(X))
        err = np.abs(self.base + self.shap(X).values.sum(axis=1) - self.predict(X))
        rows = [{"対象": "モデル出力", "式": "E[f] + Σφ = f(x)", "件数": len(X),
                 "最大絶対誤差": err_raw.max(), "許容誤差": self.tol,
                 "合格": bool((err_raw < self.tol).all())},
                {"対象": f"予測値({self.unit.label})", "式": "ベースライン + Σ寄与 = 予測値",
                 "件数": len(X), "最大絶対誤差": err.max(), "許容誤差": self.tol,
                 "合格": bool((err < self.tol).all())}]
        return pd.DataFrame(rows).set_index("対象")

    def _require(self):
        if self.X is None:
            raise RuntimeError("先に explain(X) を呼んでください")

    def row(self, key: Hashable) -> shap.Explanation:
        """explain 済みの行の SHAP(1行分)。"""
        self._require()
        return self.sv[self.X.index.get_loc(key)]

    def rows(self, keys: Iterable[Hashable]) -> shap.Explanation:
        """explain 済みの複数行の SHAP。"""
        self._require()
        return self.sv[self.X.index.get_indexer(list(keys))]

    def contributions(self, by: str = "group") -> pd.DataFrame:
        """行 × グループ(by="group")または行 × 特徴量(by="feature")の寄与。"""
        self._require()
        if by == "group":
            return self.contrib
        return pd.DataFrame(self.sv.values, index=self.X.index, columns=self.sv.feature_names)

    # ============================================================ ミクロ: 1行の分解
    def decompose(self, key: Hashable, *, label: str | None = None, name: str | None = None,
                  top: int = 3, max_display: int = 12) -> StepResult:
        """1行の分解: 分解表・日本語の要約文・ウォーターフォール図(個別特徴量 / グループ単位)。

        name を指定すると decomposition_{name}.csv と waterfall_{name}.png などを保存する。
        """
        row = self.row(key)
        label = label or key_label(key)
        actual = None if self.actual is None else float(self.actual[key])
        tbl = dc.group_decomposition_table(row, self.groups, self.unit, top, actual)
        total = tbl.loc[tbl["グループ"].str.startswith("予測値"), "寄与合計"].iloc[0]
        assert abs(total - self.pred[key]) < self.tol, "分解表の合計が予測値と一致しない"
        hide = [self.offset.name] if self.offset else []
        res = StepResult(tables={"分解表": tbl}, texts=[dc.summary_sentence(
            row, self.groups, self.unit, label, actual, hide)])
        if self._saving(name):
            save_csv(tbl, self.out.table(f"decomposition_{name}.csv"), index=False)
            extra = f"(実測 {self.unit.fmt(actual)})" if actual is not None else ""
            res.figures.append(plots.waterfall(row, f"{label}: 個別特徴量の寄与{extra}",
                                               self.out.figure(f"waterfall_{name}.png"), self.unit,
                                               max_display=max_display))
            steps = self.contrib.loc[key].reindex(
                self.contrib.loc[key].abs().sort_values(ascending=False).index)
            res.figures.append(plots.steps_waterfall(
                "ベースライン", self.base, steps, "予測値", f"{label}: グループ単位の構成{extra}",
                self.out.figure(f"group_waterfall_{name}.png"), self.unit))
        return res

    # ============================================================ 2行の差の分解
    def compare_table(self, pairs: Iterable[tuple[Hashable, Hashable]]) -> pd.DataFrame:
        """(対象, 比較対象) の組ごとの差の分解表(Δ予測値 = ΣΔ寄与。ベースラインは差で消える)。"""
        self._require()
        return dc.compare_table(self.table, self.group_names, pairs)

    def compare(self, t: Hashable, s: Hashable, *, label_t: str | None = None,
                label_s: str | None = None, title: str | None = None,
                name: str | None = None) -> StepResult:
        """2行の差を グループ別Δ寄与 に分解したウォーターフォール図と表。"""
        d = self.compare_table([(t, s)]).iloc[0]
        label_t, label_s = label_t or key_label(t), label_s or key_label(s)
        steps = d[[f"Δ{g}" for g in self.group_names]].rename(lambda c: c[1:])
        steps = steps.reindex(steps.abs().sort_values(ascending=False).index)
        res = StepResult(tables={"差の分解": d.to_frame().T})
        if self._saving(name):
            extra = (f"、Δ実測 {self.unit.fmt(d['Δ実測値'], True)}" if "Δ実測値" in d else "")
            res.figures.append(plots.steps_waterfall(
                f"予測値 {label_s}", d["予測値(比較対象)"], steps, f"予測値 {label_t}",
                title or f"{label_t} − {label_s} の要因分解(Δ予測値 "
                         f"{self.unit.fmt(d['Δ予測値'], True)}{extra})",
                self.out.figure(f"delta_{name}.png"), self.unit,
                note="ベースラインは共通なので差で消える。Δ予測値 = Σ グループ別Δ寄与。"))
        return res

    # ============================================================ マクロ
    def timeline(self, *, start=None, end=None, marker=None, marker_label: str = "",
                 title: str = "予測値の構成の推移: ベースライン + グループ別寄与",
                 name: str | None = None) -> StepResult:
        """行の並び(時系列など)に沿った ベースライン + グループ別寄与 の積み上げ図。"""
        self._require()
        ts = self.table.loc[start:end]
        res = StepResult(tables={"寄与の推移": ts})
        if self._saving(name):
            res.figures.append(plots.stacked_contributions(
                ts, self.group_names, self.out.figure(f"{name}.png"), title, self.unit,
                self.colors, marker, marker_label))
        return res

    def profile(self, keys, *, key_name: str, title: str | None = None,
                name: str | None = None, heatmap: bool = True) -> StepResult:
        """区分(季節・曜日など、行ごとのキー)ごとのグループ別平均寄与の表とヒートマップ。"""
        self._require()
        prof = dc.profile_means(self.contrib, keys, key_name)
        res = StepResult(tables={f"{key_name}別の平均寄与": prof})
        if self._saving(name):
            save_csv(prof, self.out.table(f"profile_{name}.csv"))
            if heatmap:
                res.figures.append(plots.profile_heatmap(
                    prof[self.group_names], self.out.figure(f"profile_{name}.png"),
                    title or f"{key_name}別のグループ平均寄与({self.unit.label})", self.unit))
        return res

    def overview(self, *, rows: Iterable[Hashable] | None = None, dependence: Iterable[str] = (),
                 dependence_bins: int = 8, label: str = "", name: str | None = None
                 ) -> StepResult:
        """全体傾向: 平均|SHAP|・beeswarm(rows の行)と、依存プロット(explain 済みの全行)。"""
        self._require()
        sv = self.sv if rows is None else self.rows(rows)
        head = f"{label}: " if label else ""
        imp = mean_abs_shap(sv)
        res = StepResult(tables={"平均|SHAP|": imp.rename("平均|SHAP|").to_frame()})
        partners = {}
        Xd = pd.DataFrame(self.sv.data, columns=self.sv.feature_names)
        for f in dependence:
            partners[f] = strongest_partner(self.sv, f, Xd)
            res.tables[f"{f} の値の範囲ごとの平均寄与"] = dependence_table(
                self.sv, f, n_bins=dependence_bins, unit=self.unit)
        res.texts.append("依存プロットの色(相互作用の強い特徴量): "
                         + ", ".join(f"{f}→{p}" for f, p in partners.items()))
        self.partners = partners
        if self._saving(name):
            suffix = f"_{name}" if name else ""
            save_csv(res.tables["平均|SHAP|"], self.out.table(f"mean_abs_shap{suffix}.csv"))
            res.figures.append(plots.beeswarm(sv, self.out.figure(f"beeswarm{suffix}.png"), self.unit,
                                              title=f"{head}各特徴量の寄与の分布", max_display=14))
            res.figures.append(plots.mean_abs_bar(sv, self.out.figure(f"mean_abs{suffix}.png"),
                                                  self.unit, title=f"{head}特徴量ごとの平均|寄与|"))
            for f, p in partners.items():
                save_csv(res.tables[f"{f} の値の範囲ごとの平均寄与"],
                         self.out.table(f"dependence_{f}.csv"), index=False)
                res.figures.append(plots.dependence(self.sv, f, p,
                                                    self.out.figure(f"dependence_{f}.png"), self.unit))
        return res

    # ============================================================ What-if
    def _check_scenarios(self, scenarios: dict[str, Scenario]) -> None:
        locked = set(self.offset.features) if self.offset else set()
        for sname, changes in scenarios.items():
            unknown = [f for f in changes if f not in self.X_train.columns]
            if unknown:
                raise ValueError(f"{sname}: 存在しない特徴量 {unknown}")
            bad = [f for f in changes if f in locked]
            if bad:
                raise ValueError(f"{sname}: 補正項が参照する特徴量 {bad} は動かせない")

    @produces_table
    def whatif_bounds(self, features: Iterable[str] | None = None) -> pd.DataFrame:
        """What-if で動かしてよい範囲(訓練データの分位点)と頑健σ。"""
        f = list(features) if features is not None else list(self.X_train.columns)
        return self.bounds.loc[f].join(self.sd.rename("頑健σ"))

    def _x(self, key: Hashable) -> pd.Series:
        self._require()
        return self.X.loc[key]

    def whatif(self, key: Hashable, scenarios: dict[str, Scenario], *, label: str | None = None,
               name: str | None = None, correlated_pairs: Iterable[tuple[str, str]] = (),
               max_features: int = 7) -> StepResult:
        """1行で複数の特徴量を同時に動かすシナリオ: Δ予測値(再予測)・ΔSHAP の内訳・相互作用分。

        tables: 「シナリオ別の結果」「ΔSHAP」「適用した変更」。図は ΔSHAP の上位 max_features 個 + その他。
        """
        self._check_scenarios(scenarios)
        summary, dshap, changes = scenario_analysis(
            _Predictor(self.predict), self.tree, self._x(key), self.sd, self.bounds, scenarios,
            correlated_pairs)
        assert np.allclose(summary["Δ予測値(同時)"], summary["ΣΔSHAP"], atol=self.tol)
        res = StepResult(tables={"シナリオ別の結果": summary, "ΔSHAP": dshap, "適用した変更": changes})
        if self._saving(name):
            main = dshap.abs().max().sort_values(ascending=False)
            keep = list(main[main > 1e-9].index[:max_features])
            dplot = dshap[keep].assign(その他=dshap.drop(columns=keep).sum(axis=1))
            res.figures.append(plots.scenario_dshap(
                summary, dplot, f"{label or key}: 複数の特徴量を同時に動かすシナリオ",
                self.out.figure(f"whatif_multi_{name}.png"), self.unit))
        return res

    def whatif_single(self, key: Hashable, scenarios: dict[str, Scenario], *,
                      label: str | None = None, name: str | None = None,
                      correlated_pairs: Iterable[tuple[str, str]] = ()) -> StepResult:
        """1行で特徴量を1つずつ動かしたときの Δ予測値(再予測)の表とトルネード図。"""
        self._check_scenarios(scenarios)
        summary, _, changes = scenario_analysis(
            _Predictor(self.predict), self.tree, self._x(key), self.sd, self.bounds, scenarios,
            correlated_pairs)
        df = changes.merge(summary[["シナリオ", "Δ予測値(同時)", "ΣΔSHAP", "相関注記"]], on="シナリオ")
        df = df.rename(columns={"Δ予測値(同時)": "Δ予測値"})
        df["特徴量"] = df["シナリオ"]
        df = df.reindex(df["Δ予測値"].abs().sort_values(ascending=False).index).reset_index(drop=True)
        res = StepResult(tables={"単独変更の結果": df})
        if self._saving(name):
            res.figures.append(plots.tornado(
                df, f"{label or key}: 特徴量を単独で動かしたときのΔ予測値",
                self.out.figure(f"whatif_single_{name}.png"), self.unit))
        return res

    def whatif_period(self, scenarios: dict[str, Scenario], *,
                      rows: Iterable[Hashable] | None = None, freq: str | None = None,
                      title: str = "全行にシナリオを適用したときのΔ予測値",
                      name: str | None = None) -> StepResult:
        """複数行(期間全体など)にシナリオを適用したときの総量の変化と、行別・期間別のΔ予測値。"""
        self._check_scenarios(scenarios)
        self._require()
        X = self.X if rows is None else self.X.loc[list(rows)]
        summary, daily, by_period, dshap = period_analysis(
            self.predict, lambda Z: self.shap_raw(Z).values, X, scenarios, self.sd, self.bounds,
            freq, self.tol)
        res = StepResult(tables={"シナリオ別の総量": summary, "行別のΔ予測値": daily,
                                 "特徴量別の平均ΔSHAP": dshap})
        if by_period is not None:
            res.tables["期間別の平均Δ予測値"] = by_period
        if self._saving(name):
            res.figures.append(plots.period_whatif(daily, by_period,
                                                   self.out.figure(f"{name}.png"), title, self.unit))
        return res

    # ============================================================ おまかせ実行
    @produces_table
    def auto_keys(self) -> pd.DataFrame:
        """代表行の自動選定: 予測が最大 / 最小 / 中央値付近 / (実測があれば)誤差が最大 の行。重複は次点。"""
        self._require()
        p = self.pred
        cands = {"予測が最大": p.sort_values(ascending=False).index,
                 "予測が最小": p.sort_values().index,
                 "予測が中央値付近": (p - p.median()).abs().sort_values().index}
        if self.actual is not None:
            cands["誤差が最大"] = (self.actual.reindex(p.index) - p).abs().sort_values(
                ascending=False).index
        rows, used = [], set()
        for label, order in cands.items():
            k = next(k for k in order if k not in used)
            used.add(k)
            rows.append({"区分": label, "キー": k, "予測値": p[k],
                         "実測値": None if self.actual is None else self.actual[k]})
        return pd.DataFrame(rows)

    def auto_features(self, n: int = 4) -> list[str]:
        """平均|SHAP|の大きい順に n 個の特徴量(補正項とその参照列は除く)。"""
        self._require()
        imp = mean_abs_shap(self.sv)
        locked = {self.offset.name, *self.offset.features} if self.offset else set()
        return [f for f in imp.index if f not in locked][:n]

    @staticmethod
    def auto_scenarios(features: list[str]) -> tuple[dict[str, Scenario], dict[str, Scenario]]:
        """自動シナリオ: 各特徴量を単独で +1σ(頑健σ)と、上位2つを同時に ±1σ。"""
        singles = {f"{f} +1σ": {f: ("sd", 1.0)} for f in features}
        multis = {}
        if len(features) >= 2:
            a, b = features[:2]
            multis = {f"M1 {a}&{b} +1σ": {a: ("sd", 1.0), b: ("sd", 1.0)},
                      f"M2 {a}&{b} −1σ": {a: ("sd", -1.0), b: ("sd", -1.0)}}
        return singles, multis

    def run(self, X: pd.DataFrame, actual: pd.Series | None = None, *,
            keys: Iterable[Hashable] | None = None, whatif_features: Iterable[str] | None = None,
            n_features: int = 4, freq: str | None = None, prefix: str = "auto") -> StepResult:
        """おまかせ実行: 一通りの分析をまとめて行い、out_dir があれば図表と manifest を保存する。

        1. SHAP 計算と加法性チェック            explain
        2. 代表行の分解(表・要約文・図)        decompose(keys 省略時は auto_keys で自動選定)
        3. 全体傾向(平均|SHAP|・依存プロット)  overview(寄与の大きい n_features 個)
        4. 推移と前行比の最大上昇・下落          timeline / compare(index が日時のときだけ)
        5. What-if(単独・同時・全行)            whatif_single / whatif / whatif_period

        whatif_features: What-if で動かす特徴量(省略時は寄与の大きい特徴量。動かして意味のある
                         特徴量だけに絞りたいときは必ず指定する)。freq: 全行 What-if の期間別集計(例 "M")。
        ファイル名には prefix を付ける(既存の出力と衝突させないため)。
        """
        res = StepResult()

        def add(sub: StepResult, head: str = "") -> None:
            res.tables.update({f"{head}{k}": v for k, v in sub.tables.items()})
            res.texts.extend(sub.texts)
            res.figures.extend(sub.figures)

        with MANIFEST.step(f"{prefix}: 1. SHAP計算と加法性"):
            check = self.explain(X, actual)
            res.tables["加法性チェック"] = check
            if self.out is not None:
                save_csv(check, self.out.table(f"additivity_{prefix}.csv"))
        with MANIFEST.step(f"{prefix}: 2. 代表行の分解"):
            if keys is None:
                key_df = self.auto_keys()
            else:
                keys = list(keys)
                key_df = pd.DataFrame({"区分": [f"指定{i + 1}" for i in range(len(keys))],
                                       "キー": keys})
            res.tables["代表行"] = key_df
            if self.out is not None:
                save_csv(key_df, self.out.table(f"representatives_{prefix}.csv"), index=False)
            for i, r in key_df.iterrows():
                add(self.decompose(r["キー"], label=f"{r['区分']} {key_label(r['キー'])}",
                                   name=f"{prefix}_{i + 1}"), f"{r['区分']}: ")
        feats = self.auto_features(n_features)
        with MANIFEST.step(f"{prefix}: 3. 全体傾向"):
            add(self.overview(dependence=feats, name=prefix))
        if isinstance(X.index, pd.DatetimeIndex):
            with MANIFEST.step(f"{prefix}: 4. 推移と変化の要因"):
                add(self.timeline(name=f"timeline_{prefix}"))
                diff = self.pred.diff().dropna()
                prev = dict(zip(X.index[1:], X.index[:-1]))
                for tag, t in [("up", diff.idxmax()), ("down", diff.idxmin())]:
                    add(self.compare(t, prev[t], label_t=key_label(t), label_s=key_label(prev[t]),
                                       name=f"{prefix}_{tag}"),
                        f"前行比の最大{'上昇' if tag == 'up' else '下落'}: ")
        wf = list(whatif_features) if whatif_features is not None else feats
        singles, multis = self.auto_scenarios(wf)
        with MANIFEST.step(f"{prefix}: 5. What-if"):
            for i, r in key_df.iterrows():
                head = f"{r['区分']} {key_label(r['キー'])}"
                add(self.whatif_single(r["キー"], singles, label=head, name=f"{prefix}_{i + 1}"),
                    f"{r['区分']}: ")
                if multis:
                    add(self.whatif(r["キー"], multis, label=head, name=f"{prefix}_{i + 1}"),
                        f"{r['区分']}: ")
            add(self.whatif_period(singles, freq=freq, name=f"whatif_period_{prefix}"), "全行: ")
        if self.out is not None:
            self.save_manifest()
        res.texts.insert(0, f"寄与の大きい特徴量: {', '.join(feats)} / What-if で動かした特徴量: "
                            f"{', '.join(wf)}(±1σ = 訓練データの頑健σ)")
        return res

    # ============================================================ その他
    def _saving(self, name: str | None) -> bool:
        return self.out is not None and name is not None

    def save_manifest(self, title: str = "出力ファイルと生成I/Fの対応") -> tuple[Path, Path]:
        """out_dir/manifest.csv と manifest.md(出力ファイル → 生成 I/F の対応)を書き出す。"""
        if self.out is None:
            raise RuntimeError("out_dir を指定していないので保存先が無い")
        return MANIFEST.save(self.out.root, title)

    @staticmethod
    def linear_check(linear_model, X_train: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
        """線形モデルで SHAP = 係数 × (x − 訓練平均) となるかの検証表(SHAP 実装の確認用)。"""
        return linear_shap_check(linear_model, X_train, X_test)

    @staticmethod
    def group_sum(sv: shap.Explanation, index: pd.Index,
                  groups: dict[str, list[str]]) -> pd.DataFrame:
        """任意の SHAP(他モデルのものなど)を行 × グループの寄与に集約する。"""
        return dc.group_frame(sv, index, groups)

    @staticmethod
    def feature_table(descriptions: dict[str, str],
                      groups: dict[str, list[str]] | None = None) -> pd.DataFrame:
        """特徴量の説明表(groups を渡すとグループ列も付ける)。"""
        df = description_table(descriptions)
        if groups:
            g = pd.DataFrame([{"特徴量": f, "グループ": k} for k, fs in groups.items() for f in fs])
            df = df.merge(g, on="特徴量")
        return df


def quick_analyze(model, X_train: pd.DataFrame, X: pd.DataFrame,
                  actual: pd.Series | None = None, *, out_dir: str | Path | None = "outputs",
                  unit: Unit | None = None, groups: dict[str, list[str]] | None = None,
                  offset: Offset | None = None, **run_kwargs) -> tuple[ShapAnalyzer, StepResult]:
    """1行で一通りの分析を行う: ShapAnalyzer を作って run を呼ぶ。

        az, res = quick_analyze(model, X_train, X_test, y_test)
        res.show()

    run_kwargs は ShapAnalyzer.run に渡す(keys / whatif_features / n_features / freq / prefix)。
    返り値の ShapAnalyzer で、続けて個別のメソッドも呼べる。
    """
    az = ShapAnalyzer(model, X_train, unit=unit, groups=groups, offset=offset, out_dir=out_dir)
    return az, az.run(X, actual, **run_kwargs)
