"""LDA (Latent Dirichlet Allocation) によるトピックモデリングのサンプル。

LDAは各文書を単語の出現頻度(Bag-of-Words)で表現し、
「文書はいくつかのトピックの混合で構成され、各トピックは特定の単語が
出現しやすい確率分布である」という生成モデルを仮定してトピックを推定する
古典的な手法。単語の"表記"の共起パターンだけを見るため、
同じ意味でも異なる単語(例: "movie" と "film")は別々の情報として扱われる。

実行方法:
    python lda_sample.py
"""

from __future__ import annotations

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from data import load_sample_documents

N_TOPICS = 5
N_TOP_WORDS = 8
RANDOM_STATE = 42


def build_lda_model(
    documents: list[str],
    n_topics: int = N_TOPICS,
    random_state: int = RANDOM_STATE,
):
    """Bag-of-Words + LDA でトピックモデルを学習する。

    Returns:
        lda: 学習済み LatentDirichletAllocation
        vectorizer: 学習済み CountVectorizer(語彙の参照に使う)
        doc_topic_matrix: 文書ごとのトピック混合比率 (n_docs, n_topics)
    """
    vectorizer = CountVectorizer(stop_words="english", min_df=2)
    doc_term_matrix = vectorizer.fit_transform(documents)

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=random_state,
        learning_method="batch",
        max_iter=50,
    )
    doc_topic_matrix = lda.fit_transform(doc_term_matrix)
    return lda, vectorizer, doc_topic_matrix


def get_top_words_per_topic(lda, vectorizer, n_top_words: int = N_TOP_WORDS) -> dict[int, list[str]]:
    """各トピックの代表単語(出現確率が高い単語)を取得する。"""
    feature_names = vectorizer.get_feature_names_out()
    top_words_per_topic = {}
    for topic_idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[::-1][:n_top_words]
        top_words_per_topic[topic_idx] = [feature_names[i] for i in top_indices]
    return top_words_per_topic


def main() -> None:
    documents, labels = load_sample_documents()
    lda, vectorizer, doc_topic_matrix = build_lda_model(documents)
    top_words_per_topic = get_top_words_per_topic(lda, vectorizer)

    print("=== LDA: 各トピックの代表単語 ===")
    for topic_idx, words in top_words_per_topic.items():
        print(f"トピック{topic_idx}: {', '.join(words)}")

    print("\n=== LDA: 各文書の主トピック (先頭10件) ===")
    for doc, label, topic_dist in zip(documents[:10], labels[:10], doc_topic_matrix[:10]):
        main_topic = int(topic_dist.argmax())
        confidence = topic_dist[main_topic]
        print(f"[正解:{label:10s}] -> トピック{main_topic} (確信度 {confidence:.2f}) : {doc}")


if __name__ == "__main__":
    main()
