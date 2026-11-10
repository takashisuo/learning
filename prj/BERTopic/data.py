"""BERTopic と LDA の比較用サンプルデータ。

20newsgroups のような有名データセットはネットワーク経由でのダウンロードが
必要になり、実行環境によっては失敗する。ここではオフラインでも確実に再現できる
よう、5つの明確なトピック(スポーツ／テクノロジー／料理／金融／映画)を持つ
短い英文ドキュメント集合を自作する。

英語にしているのは、CountVectorizer(stop_words="english")や BERTopic の
デフォルト埋め込みモデルが英語のストップワード・語彙を前提にしており、
そのままの設定で分かりやすく比較できるため。
"""

from __future__ import annotations

# トピックごとに短い英文ドキュメントを用意する。
# キー: 正解トピック名(評価用のラベルとしてのみ使い、モデルには渡さない)
_TOPIC_DOCUMENTS: dict[str, list[str]] = {
    "sports": [
        "The soccer team won the championship after a thrilling final match.",
        "She scored three goals in the basketball game last night.",
        "The marathon runner trained for months before the big race.",
        "Our baseball team hit a home run in the final inning to win.",
        "The tennis player served an ace to win the championship point.",
        "The coach praised the team's defense after the football victory.",
        "He broke the world record in the 100 meter sprint at the olympics.",
        "The hockey match ended in a dramatic overtime goal.",
        "The swimming team qualified for the national championship finals.",
        "Fans cheered loudly as the striker scored the winning goal.",
        "The volleyball team practiced serves and spikes before the tournament.",
        "The cyclist won the mountain stage of the famous bicycle race.",
        "The boxer trained hard for the upcoming championship fight.",
        "The golfer sank a long putt to win the tournament on the final hole.",
        "The rugby team scored a try in the last minute of the match.",
    ],
    "technology": [
        "The new smartphone features a faster processor and better camera.",
        "Researchers trained a deep learning model to recognize images.",
        "The software update fixed several bugs and improved performance.",
        "Engineers developed a new algorithm to compress video files.",
        "The startup released an app that uses artificial intelligence.",
        "The company launched a cloud computing platform for developers.",
        "A new chip design promises to make laptops run much faster.",
        "The robot uses sensors and machine learning to navigate rooms.",
        "Developers use version control systems to manage source code.",
        "The data center relies on servers to store and process information.",
        "Scientists built a neural network to translate languages automatically.",
        "The programmer debugged the code and deployed the new feature.",
        "The website uses a database to store user information securely.",
        "Engineers are testing self driving cars on public roads.",
        "The new operating system update adds better battery management.",
    ],
    "cooking": [
        "The chef prepared a delicious pasta dish with fresh tomatoes and basil.",
        "She baked a chocolate cake for the birthday party.",
        "The recipe calls for garlic, onions, and olive oil.",
        "He grilled chicken with a spicy marinade for dinner.",
        "The bakery sells fresh bread and pastries every morning.",
        "The soup simmered on the stove with herbs and vegetables.",
        "The restaurant serves sushi made with fresh fish and rice.",
        "She whisked eggs and sugar to make a light and fluffy sponge cake.",
        "The recipe recommends marinating the beef overnight for extra flavor.",
        "He seasoned the steak with salt, pepper, and rosemary before grilling.",
        "The kitchen smelled of freshly baked cookies and cinnamon.",
        "The cook chopped vegetables and added them to the stir fry.",
        "The salad was made with lettuce, tomatoes, and a lemon vinaigrette.",
        "The pastry chef decorated the cake with fresh strawberries and cream.",
        "The stew needs to simmer for two hours to become tender.",
    ],
    "finance": [
        "The stock market rallied after the central bank cut interest rates.",
        "Investors bought shares of the company after strong earnings.",
        "The company reported record profits in the fourth quarter.",
        "The bank raised interest rates to control inflation.",
        "Analysts expect the currency to weaken against the dollar.",
        "The fund manager diversified the portfolio across several sectors.",
        "The startup raised millions of dollars in a new funding round.",
        "The government announced a new budget to reduce the deficit.",
        "Shareholders approved the merger between the two companies.",
        "The economy grew steadily despite concerns about inflation.",
        "The investor sold bonds and bought more stocks for higher returns.",
        "The company's stock price dropped after disappointing sales figures.",
        "The central bank monitors inflation to set monetary policy.",
        "The trader analyzed market trends before making an investment.",
        "The insurance company reported higher premiums this year.",
    ],
    "movies": [
        "The actor won an award for his performance in the new drama film.",
        "The director released a highly anticipated sequel this summer.",
        "Critics praised the movie for its stunning visual effects.",
        "The film premiered at the festival to a standing ovation.",
        "The actress starred in a thriller that topped the box office.",
        "The studio announced a new superhero movie for next year.",
        "The soundtrack of the film received praise from music critics.",
        "The screenplay was adapted from a bestselling novel.",
        "The comedy film made audiences laugh throughout the screening.",
        "The animated movie became a huge hit with children and families.",
        "The cast reunited for a special screening of the classic film.",
        "The documentary explored the life of a famous musician.",
        "The horror movie kept audiences on the edge of their seats.",
        "The film festival showcased independent movies from around the world.",
        "The actor prepared for months to play the historical figure.",
    ],
}


def load_sample_documents() -> tuple[list[str], list[str]]:
    """サンプルドキュメントと正解ラベルを返す。

    ラベルはあくまで人間が結果を確認するための参考情報であり、
    LDA/BERTopic のどちらの学習にも使用しない(教師なし学習のため)。

    Returns:
        documents: 文書(文字列)のリスト
        labels: 各文書の正解トピック名のリスト(documents と同じ順序・長さ)
    """
    documents: list[str] = []
    labels: list[str] = []
    for topic_name, docs in _TOPIC_DOCUMENTS.items():
        documents.extend(docs)
        labels.extend([topic_name] * len(docs))
    return documents, labels
