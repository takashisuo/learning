# BERTopic と LDA の違いサンプル

トピックモデリングの代表的な2手法、LDA(Latent Dirichlet Allocation)と
BERTopic を同じサンプルデータで動かして、アプローチの違いを比較するサンプル実装。

## LDA と BERTopic の違い

| | LDA | BERTopic |
|---|---|---|
| 文書の表現 | 単語の出現頻度(Bag-of-Words) | Sentence-Transformers による埋め込みベクトル(意味表現) |
| トピック抽出の仕組み | 確率的生成モデル(文書=トピックの混合、トピック=単語分布) | 埋め込み → UMAPで次元削減 → HDBSCANでクラスタリング → c-TF-IDFで代表語抽出 |
| 捉えられる関係 | 単語の"表記"の共起パターン | 文の"意味"の近さ |
| トピック数の指定 | 事前に `n_components` で指定する必要がある | HDBSCANが自動でクラスタ数を決める(`min_topic_size` で調整可能) |
| 分類できない文書 | 必ずどれかのトピックに割り当てられる | 「外れ値」(トピックID `-1`)として弾かれることがある |
| 弱点 | 同義語や言い換えに弱い(語彙が一致しないと同じトピックにまとまりにくい) | 埋め込みモデルの計算コストが高く、小規模データではクラスタが安定しないことがある |

## ファイル構成

| ファイル | 内容 |
|---|---|
| `data.py` | サンプルデータ生成(`load_sample_documents`)。ネットワーク不要でオフラインで再現できるよう、5トピック(sports/technology/cooking/finance/movies)× 15件の英文ドキュメントを自作している |
| `lda_sample.py` | LDA単体のサンプル。`CountVectorizer` + `LatentDirichletAllocation` でトピックを推定 |
| `bertopic_sample.py` | BERTopic単体のサンプル。`UMAP` の `random_state` を固定して結果を再現可能にしている |
| `main.py` | 上記2つを実行し、正解ラベルとの対応(クロス集計)や代表単語を並べて比較するエントリーポイント。実行すると `output/bertopic_topics.html` にBERTopicのトピック分布を出力する |

## 実行方法

```bash
pip install -r requirements.txt

# LDAのみ
python lda_sample.py

# BERTopicのみ
python bertopic_sample.py

# 両方を実行して比較
python main.py
```

BERTopicは初回実行時に Sentence-Transformers の埋め込みモデル
(`all-MiniLM-L6-v2`)をダウンロードするため、インターネット接続が必要。
LDA(`lda_sample.py`)はネットワーク接続不要で動作する。

## 実行結果の見方

`main.py` を実行すると、各モデルについて

1. 抽出されたトピックの代表単語
2. 正解ラベル(sports/technology/...)× 割当トピックID のクロス集計表

が表示される。理想的には各正解ラベルがほぼ1つのトピックIDに集中する。
LDAとBERTopicでこの集計表を見比べると、同じデータでもトピックの切り方
(境界の引き方)が手法によって変わることを確認できる。
