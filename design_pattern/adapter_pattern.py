"""
Adapter パターン（構造パターン）
================================

■ 目的
    インターフェース（メソッド名・引数・戻り値の形）が異なる既存のクラス
    （特に古い社内ライブラリやライブラリ間の差異）を、自分のコードが期待する
    共通インターフェースに変換する「翻訳役」のパターン。

■ いつ使う？
    - 複数のモデル実装（scikit-learn製、自前実装、社内の古いコード等）を
      同じ評価パイプラインに混在させたい場合
    - 既存コードを変更できない（社内の古いモデルコードなど）が、
      自分のシステムのインターフェースに合わせたい場合

■ 機械学習プロジェクトでの使いどころ
    - scikit-learn の推定器は `fit(X, y)` / `predict(X)` という共通の作法を
      持つが、社内の昔ながらのルールベースモデルや、別チームが作った
      独自実装は `train_model()` / `infer()` のように異なる名前で
      作られていることが多い。
    - これらをAdapterで共通インターフェース（Classifier）に揃えることで、
      モデルの種類を問わず同じ評価コード（accuracy計算など）を再利用できる。

■ 動作確認
    Python 3.11.13 (conda env: samurai) / scikit-learn 1.5.1
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------
# 1. 既存の（インターフェースが揃っていない）モデル実装たち
# ---------------------------------------------------------------
class LegacyRuleBasedModel:
    """社内に昔からある、ルールベースの分類ロジック。

    scikit-learnとは無関係に作られているため、メソッド名が独自（train_model/infer）。
    """

    def __init__(self, threshold: float) -> None:
        self._threshold = threshold

    def train_model(self, X: np.ndarray, y: np.ndarray) -> None:
        # ルールベースのため学習データから「しきい値に使う特徴量の列」だけ覚える
        self._feature_index = 2  # iris では花びらの長さに相当する列

    def infer(self, X: np.ndarray) -> List[int]:
        return [1 if row[self._feature_index] > self._threshold else 0 for row in X]


class InHouseGradientModel:
    """別チームが独自実装した勾出（勾配ブースティング風）モデルを模したクラス。

    こちらも独自のメソッド名（fit_model/predict_proba_custom）を持つ想定。
    """

    def __init__(self) -> None:
        self._backbone = LogisticRegression(max_iter=200)

    def fit_model(self, X: np.ndarray, y: np.ndarray) -> None:
        self._backbone.fit(X, y)

    def predict_proba_custom(self, X: np.ndarray) -> np.ndarray:
        return self._backbone.predict(X)


# ---------------------------------------------------------------
# 2. アプリ側（評価パイプライン）が使いたい共通インターフェース
# ---------------------------------------------------------------
class Classifier(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


# ---------------------------------------------------------------
# 3. 各モデルをClassifierインターフェースに変換するAdapter
# ---------------------------------------------------------------
class LegacyRuleBasedAdapter(Classifier):
    def __init__(self, legacy_model: LegacyRuleBasedModel) -> None:
        self._legacy_model = legacy_model

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._legacy_model.train_model(X, y)  # train_model -> fit に変換

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array(self._legacy_model.infer(X))  # infer -> predict に変換


class InHouseGradientAdapter(Classifier):
    def __init__(self, in_house_model: InHouseGradientModel) -> None:
        self._in_house_model = in_house_model

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._in_house_model.fit_model(X, y)  # fit_model -> fit に変換

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._in_house_model.predict_proba_custom(X)  # predict_proba_custom -> predict に変換


def evaluate(model: Classifier, X_test: np.ndarray, y_test: np.ndarray, name: str) -> None:
    """共通インターフェース(Classifier)だけに依存する評価関数。

    Adapterのおかげで、モデルの出自（scikit-learn/自前/社内レガシー）を問わず
    このまま1つの関数で評価できる。
    """
    predictions = model.predict(X_test)
    accuracy = float(np.mean(predictions == y_test))
    print(f"[{name}] accuracy={accuracy:.3f}")


if __name__ == "__main__":
    X, y = load_iris(return_X_y=True)
    # ルールベース判定を二値問題に単純化するため、ラベルを「0か / それ以外か」に変換
    y_binary = (y != 0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.3, random_state=0)

    classifiers = [
        ("LegacyRuleBasedModel", LegacyRuleBasedAdapter(LegacyRuleBasedModel(threshold=2.0))),
        ("InHouseGradientModel", InHouseGradientAdapter(InHouseGradientModel())),
    ]

    for name, adapted_model in classifiers:
        adapted_model.fit(X_train, y_train)
        evaluate(adapted_model, X_test, y_test, name)
