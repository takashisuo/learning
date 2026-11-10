"""LDA と BERTopic の違いをサンプルコードで比較するエントリーポイント。

同じサンプルデータ(data.py)に対して LDA と BERTopic それぞれでトピック
モデリングを行い、
    - 抽出されたトピックの代表単語
    - 各文書がどのトピックに分類されたか
    - 正解トピック(ラベル)ごとの分類のばらつき
を並べて比較する。

実行方法:
    python main.py
実行すると output/bertopic_topics.html にBERTopicのトピック分布の
可視化(棒グラフ)を出力する。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bertopic_sample import build_bertopic_model
from data import load_sample_documents
from lda_sample import build_lda_model, get_top_words_per_topic

OUTPUT_DIR = Path(__file__).parent / "output"


def summarize_lda(documents, labels):
    lda, vectorizer, doc_topic_matrix = build_lda_model(documents)
    top_words_per_topic = get_top_words_per_topic(lda, vectorizer)
    assigned_topics = doc_topic_matrix.argmax(axis=1)

    print("--- LDA のトピック(代表単語) ---")
    for topic_idx, words in top_words_per_topic.items():
        print(f"  トピック{topic_idx}: {', '.join(words)}")

    return assigned_topics


def summarize_bertopic(documents, labels):
    topic_model, topics = build_bertopic_model(documents)

    print("--- BERTopic のトピック(代表単語) ---")
    for topic_id in sorted(set(topics)):
        if topic_id == -1:
            words = ["(外れ値クラスタ)"]
        else:
            words = [word for word, _ in topic_model.get_topic(topic_id)[:8]]
        print(f"  トピック{topic_id}: {', '.join(words)}")

    return topic_model, topics


def print_crosstab(title, labels, assigned_topics):
    """正解ラベル x 割当トピック のクロス集計を表示する。

    理想的には各正解ラベル(sports/technology/...)が、ほぼ1つの
    トピックIDに集中しているはず。複数のトピックIDに分散していたり、
    別ラベルと同じトピックに混ざっていたりすると、そのモデルが
    このデータではトピックをうまく分離できていないことを示す。
    """
    df = pd.DataFrame({"正解ラベル": labels, "割当トピック": assigned_topics})
    crosstab = pd.crosstab(df["正解ラベル"], df["割当トピック"])
    print(f"\n--- {title}: 正解ラベル x 割当トピック ---")
    print(crosstab.to_string())


def main() -> None:
    documents, labels = load_sample_documents()

    print("=" * 60)
    print("LDA と BERTopic の比較")
    print(f"サンプル文書数: {len(documents)} (5トピック x 各15件)")
    print("=" * 60)

    print("\n[1] LDA ".ljust(60, "-"))
    lda_assigned_topics = summarize_lda(documents, labels)
    print_crosstab("LDA", labels, lda_assigned_topics)

    print("\n[2] BERTopic ".ljust(60, "-"))
    topic_model, bertopic_assigned_topics = summarize_bertopic(documents, labels)
    print_crosstab("BERTopic", labels, bertopic_assigned_topics)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "bertopic_topics.html"
    topic_model.visualize_barchart(top_n_topics=len(set(bertopic_assigned_topics))).write_html(str(out_path))
    print(f"\nBERTopicのトピック分布を可視化しました: {out_path.resolve()}")

    print("\n" + "=" * 60)
    print("まとめ:")
    print("- LDAは単語の出現頻度(Bag-of-Words)に基づき、確率モデルとしてトピックを推定する。")
    print("  同じ単語をよく共有するか否かで文書をまとめるため、語彙が被らないと分離しにくい。")
    print("- BERTopicは文の意味を捉えた埋め込みベクトルでクラスタリングするため、")
    print("  表記が異なっても意味の近い文書を同じトピックにまとめやすい。")
    print("  一方で外れ値(トピック-1)として弾かれる文書が出ることがある。")
    print("=" * 60)


if __name__ == "__main__":
    main()
