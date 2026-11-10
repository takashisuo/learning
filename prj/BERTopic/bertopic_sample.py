"""BERTopic によるトピックモデリングのサンプル。

BERTopic は次の3ステップでトピックを抽出する。
    1. Sentence-Transformers で各文書を意味を捉えた埋め込みベクトルに変換する
    2. UMAP で埋め込みベクトルを低次元に圧縮する
    3. HDBSCAN で似た文書をクラスタリングし、クラスタごとに c-TF-IDF で
       代表単語を抽出する

LDA が単語の"表記"の共起パターンを見るのに対し、BERTopic は文の"意味"の
近さでクラスタリングするため、単語が完全に一致しなくても意味的に近い文書を
同じトピックにまとめられるのが特徴。

実行方法:
    python bertopic_sample.py

初回実行時は Sentence-Transformers の埋め込みモデル(all-MiniLM-L6-v2)を
ダウンロードするため、インターネット接続が必要。
"""

from __future__ import annotations

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from data import load_sample_documents

MIN_TOPIC_SIZE = 5
RANDOM_STATE = 42


def build_bertopic_model(documents: list[str], min_topic_size: int = MIN_TOPIC_SIZE):
    """埋め込み -> UMAP -> HDBSCAN -> c-TF-IDF の流れでトピックモデルを学習する。

    UMAP に random_state を固定して渡すことで、結果を再現可能にしている
    (BERTopic はデフォルトのままだと乱数の影響で実行のたびに結果が変わる)。

    Returns:
        topic_model: 学習済み BERTopic
        topics: 各文書に割り当てられたトピックID(-1は外れ値扱い)
    """
    umap_model = UMAP(
        n_neighbors=10,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=RANDOM_STATE,
    )
    # c-TF-IDF で代表単語を抽出する際にストップワードを除去し、キーワードを見やすくする
    # (クラスタリング自体は埋め込みベクトルで行うため、意味的な分類には影響しない)
    vectorizer_model = CountVectorizer(stop_words="english")
    topic_model = BERTopic(
        umap_model=umap_model,
        vectorizer_model=vectorizer_model,
        min_topic_size=min_topic_size,
        calculate_probabilities=False,
        verbose=False,
    )
    topics, _ = topic_model.fit_transform(documents)
    return topic_model, topics


def main() -> None:
    documents, labels = load_sample_documents()
    topic_model, topics = build_bertopic_model(documents)

    print("=== BERTopic: 抽出されたトピック一覧 ===")
    topic_info = topic_model.get_topic_info()
    print(topic_info.to_string(index=False))

    print("\n=== BERTopic: 各文書の割当トピック (先頭10件) ===")
    for doc, label, topic_id in zip(documents[:10], labels[:10], topics[:10]):
        if topic_id == -1:
            top_words = ["(外れ値)"]
        else:
            top_words = [word for word, _ in topic_model.get_topic(topic_id)[:5]]
        print(f"[正解:{label:10s}] -> トピック{topic_id} {top_words} : {doc}")


if __name__ == "__main__":
    main()
