# 時系列モデル + Conformal Prediction サンプル

機械学習モデル(scikit-learn)の点予測に対して、Split Conformal Prediction で
疑似的な予測区間(信頼区間相当)を付与するサンプル実装。

## やっていること

1. トレンド + 週次季節性 + ノイズを持つサンプル時系列データを生成
2. ラグ・移動平均などの特徴量を作成
3. train / calibration / test に時系列順で分割
4. train でモデルを学習し、calibration データの残差から予測区間の幅(`q_hat`)を求める
5. test データに点予測 ± `q_hat` で予測区間を付け、plotly で可視化

## ファイル構成

| ファイル | 内容 |
|---|---|
| `data.py` | サンプルデータ生成 (`generate_sample_timeseries`) と特徴量作成 (`create_features`) |
| `conformal.py` | Split Conformal Prediction 本体。`SplitConformalRegressor` クラス(`fit` → `calibrate` → `predict`) |
| `models.py` | `TrendResidualRegressor`。GBMなど木ベースのモデルは学習範囲外(未来)への外挿が苦手なので、トレンド成分を線形回帰で分離し、残差だけを木モデルに学習させるラッパー |
| `main.py` | 上記を組み合わせて実行するエントリーポイント。実行すると `output/conformal_prediction.html` を出力 |

## 実行方法

```bash
pip install -r requirements.txt
python main.py
```

コンソールに実測被覆率(目標90%に対して実際どれくらいの割合で真値が区間に入ったか)が
表示され、`output/conformal_prediction.html` に実測値・点予測・予測区間帯を重ねた
グラフが出力される。

## Split Conformal Prediction のポイント

- 「予測 ± 一定幅」というシンプルな形で予測区間を作る手法
- 幅(`q_hat`)は calibration データでの残差(|実測 - 予測|)の分位点から決める
- 分位点の取り方に有限サンプル補正 (`ceil((n+1)(1-alpha))/n`) が入っており、
  データ数が有限でも理論的に目標被覆率以上を達成できるのが特徴
- 前提はデータの交換可能性(exchangeability)。時系列でトレンドがあると
  calibration と test で残差の分布がズレやすく、被覆率が目標から外れる原因になる
  (今回 `TrendResidualRegressor` を入れているのはこれの対策)

## use_library/

この自作実装を MAPIE ライブラリ(`SplitConformalRegressor` / `TimeSeriesRegressor`)と
比較するサンプルを [use_library/](use_library/) に置いている。詳細は
[use_library/README.md](use_library/README.md) を参照。
