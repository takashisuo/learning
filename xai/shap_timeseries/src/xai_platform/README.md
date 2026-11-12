# xai_platform

木モデル(LightGBM・XGBoost・scikit-learn の木など)の予測値を、SHAP で
**「ベースライン + 特徴量(グループ)ごとの寄与 = 予測値」** に分解するパッケージ。

入口は `ShapAnalyzer` クラス1つだけ(Facade)。内部の `_core/` を直接使う必要はない。

## いちばん簡単な使い方(おまかせ実行)

```python
from xai_platform import quick_analyze, Unit

az, res = quick_analyze(model, X_train, X_test, y_test,          # y_test(実測値)は省略可
                        unit=Unit("件/日", digits=0), out_dir="outputs",
                        whatif_features=["temp_c", "hum_pct"])   # What-if で動かす特徴量(推奨)
res.show()                                                       # ノートブックで結果を一覧表示
```

これだけで次を一通り行い、`out_dir` に図表と `manifest.md` を保存する(ファイル名には `auto` が付く)。

1. SHAP 計算と加法性チェック
2. 代表行の分解(自動選定: 予測が最大 / 最小 / 中央値付近 / 誤差が最大。`keys=[...]` で指定も可)
3. 全体傾向(平均|SHAP|・beeswarm・寄与の大きい特徴量の依存プロット)
4. index が日時なら、寄与の推移と、前の行からの最大上昇・最大下落の要因分解
5. What-if(各特徴量を単独で +1σ、上位2つを同時に ±1σ、全行に +1σ)

注意:
- `whatif_features` を省略すると、寄与の大きい特徴量を自動で動かす。時系列のラグのように
  「動かすと現実と矛盾する」特徴量が選ばれることがあるので、動かして意味のある特徴量を指定するのが安全。
- 作成済みの `ShapAnalyzer` なら `az.run(X_test, y_test, ...)` でも同じ。実行後は、下の個別メソッドで
  深掘りできる。

## 最小の使い方(個別メソッド)

```python
from xai_platform import ShapAnalyzer, Unit

az = ShapAnalyzer(model, X_train,
                  unit=Unit("件/日", digits=0),                         # 目的変数の単位と表示桁数
                  groups={"天気": ["temp_c", "hum_pct"], "暦": ["weekday"]},  # 漏れた特徴量は「その他」
                  out_dir="outputs")                                   # 図表の保存先(省略すると保存しない)

az.explain(X_test, actual=y_test)       # SHAP 計算 + 加法性チェック(最初に必ず呼ぶ)

az.decompose(key, name="day1").show()   # 1行の分解表・要約文・ウォーターフォール図
az.compare(t, s, name="t_vs_s")         # 2行の差の分解(Δ予測値 = ΣΔ寄与)
az.timeline(name="timeline")            # 行の並び(時系列)に沿った寄与の積み上げ図
az.profile(keys, key_name="曜日", name="weekday")   # 区分ごとの平均寄与 + ヒートマップ
az.overview(dependence=["temp_c"], name="all")      # 平均|SHAP|・beeswarm・依存プロット

scen = {"気温+5℃": {"temp_c": ("abs", 5)}, "湿度−10%": {"hum_pct": ("pct", -0.10)}}
az.whatif_single(key, scen, name="day1")     # 1つずつ動かしたときの Δ予測値(トルネード図)
az.whatif(key, scen, name="day1")            # 同時に動かしたときの Δ予測値・ΔSHAP・相互作用分
az.whatif_period(scen, freq="M", name="all") # 全行に適用したときの総量の変化

az.save_manifest()                      # 出力ファイルと、それを作ったメソッドの対応表
```

- `name` を渡したときだけ図表をファイルに保存する。戻り値はすべて `StepResult`
  (`.tables` / `.figures` / `.texts`。ノートブックでは `.show()` で表示)。
- `key` は `explain` に渡した `X` の index のラベル。
- シナリオの変更は `(種類, 量)`。種類は `"abs"`(元の単位で加算)/ `"pct"`(現在値の割合)/
  `"sd"`(訓練データの頑健σの倍数)。動かす範囲は訓練データの 1〜99 パーセンタイルに制限する
  (範囲外はクリップして警告)。

## 補正項(Offset): モデルの外で予測値に足す値がある場合

目的変数を「直近平均からの差」などに変換して学習した場合は、予測値 = モデル出力 + 直近平均 になる。
その「足す値」を `Offset` で渡すと、元の単位のまま分解できる。

```python
from xai_platform import Offset

az = ShapAnalyzer(model, X_train,
                  offset=Offset(name="直近水準", func=lambda X: X["roll7_mean"],
                                features=("roll7_mean",), group="自己回帰"))
```

- 補正項は「訓練データでの平均」(ベースラインに入る)と「そこからの差」(`name` の寄与)に分かれる。
- `features` に挙げた列は What-if で動かせない(動かすと Δ予測値 = ΣΔSHAP が崩れるため)。
- 対数変換のように足し算で戻せない変換には使えない。

## 主なメソッド

| メソッド | 内容 | 保存するもの(`name` 指定時) |
|---|---|---|
| `run(X, actual=None, ...)` | おまかせ実行(上記 1〜5) | 各メソッドの出力 + `additivity_auto.csv`、`representatives_auto.csv`、`manifest.*` |
| `explain(X, actual=None)` | SHAP 計算・加法性チェック。以降の前提 | なし |
| `additivity(X)` | 加法性チェックの表 | なし |
| `decompose(key)` | 1行の分解表・要約文 | `decomposition_*.csv`、`waterfall_*.png`、`group_waterfall_*.png` |
| `compare(t, s)` / `compare_table(pairs)` | 2行の差の分解 | `delta_*.png` |
| `timeline()` | 寄与の積み上げ推移 | `{name}.png` |
| `profile(keys, key_name=...)` | 区分ごとの平均寄与 | `profile_*.csv`、`profile_*.png` |
| `overview(rows=None, dependence=[...])` | 全体傾向 | `mean_abs_shap_*.csv`、`beeswarm_*.png`、`mean_abs_*.png`、`dependence_*.csv/png` |
| `whatif_single` / `whatif` / `whatif_period` | What-if | `whatif_single_*.png`、`whatif_multi_*.png`、`{name}.png` |
| `contributions(by="group"/"feature")` | 行 × グループ(特徴量)の寄与 | なし |
| `predict` / `shap` / `row(key)` | 予測値・SHAP 値の取得 | なし |
| `linear_check`(静的) | 線形モデルで SHAP = 係数 ×(x − 平均)の検証 | なし |
| `feature_table` / `group_sum`(静的) | 特徴量の説明表 / 任意の SHAP のグループ集約 | なし |

## 自前の図表を保存するとき

```python
from xai_platform import analysis_step, save_csv, save_figure, plt

with analysis_step("5. 追加分析"):          # manifest に記録する分析ステップ名
    save_csv(df, "outputs/tables/extra.csv", source="アプリ側で作成")
```

## 構成

```
xai_platform/
├── __init__.py   公開するもの(ShapAnalyzer・Offset・Unit・保存の補助)
├── analyzer.py   ShapAnalyzer(Facade)
├── README.md
└── _core/        内部実装(units / io / provenance / explain / decompose / whatif / plots)
```
