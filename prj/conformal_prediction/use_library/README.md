# MAPIE ライブラリとの比較サンプル

親フォルダで自作した Split Conformal Prediction と、[MAPIE](https://mapie.readthedocs.io/)
ライブラリの2クラスを、同じデータ・同じベースモデル(`TrendResidualRegressor`)で比較する。

- `SplitConformalRegressor` : 通常のクロスセクショナル(i.i.d.に近い)データ向け
- `TimeSeriesRegressor` (`method="enbpi"`) : 時系列データ向け。EnbPI はブートストラップで
  複数モデルを学習し、そのアンサンブルの Out-of-Bag 残差から区間を求める手法

## 実行方法

```bash
pip install -r requirements.txt
python main.py
```

被覆率・平均区間幅を3手法分コンソールに表示し、`output/mapie_comparison.html` に
実測値と各手法の予測区間を重ねたグラフを出力する。

## 結果の傾向

手元の実行では、被覆率はどの手法もほぼ目標(90%)通りだったが、区間の幅は
`TimeSeriesRegressor` (EnbPI) が最も狭くなった(自作/`SplitConformalRegressor` は
残差の分位点1つだけで幅を決めるのに対し、EnbPI は複数モデルのアンサンブルを使う分、
同じ被覆率をより狭い区間で達成しやすい)。

- 自作 / MAPIE の `SplitConformalRegressor` はロジックがほぼ同じなので結果もほぼ一致する
  → 「自分で実装した内容が、ライブラリの一般的な実装と整合している」ことの確認にもなる
- 時系列データでは `TimeSeriesRegressor` (EnbPI) の方が区間が効率的(狭い)になりやすい
  → ユーザーの仮説通り、Mapie(Split系)は i.i.d. 的なデータ、
    MapieTimeSeriesRegressor は時系列データに向いている、という傾向が確認できる
