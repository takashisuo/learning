"""
Strategy パターン（戦略パターン）
================================

■ 目的
    「アルゴリズム（処理方法）」をクラスとして切り出し、実行時に自由に
    差し替えられるようにするパターン。
    呼び出し側のコード（ModelTrainer）は「どのアルゴリズムを使っているか」
    を知らなくてよい。

■ いつ使う？
    - 同じ目的（例: 分類モデルの学習）に対して、複数のアルゴリズムがあり、
      状況によって切り替えたい場合。
    - if/elif でアルゴリズム選択ロジックが増えてきて見通しが悪くなってきた場合。

■ 機械学習プロジェクトでの使いどころ
    - 分類アルゴリズム（ロジスティック回帰 / ランダムフォレスト / k近傍法）を
      同じ学習・評価コードのまま切り替える。
    - 実は scikit-learn の Estimator API（`fit(X, y)` / `predict(X)`）自体が
      Strategyパターンの実例になっている。本サンプルではそれをもう一段
      ラップして、「アルゴリズムの切り替え」がコードのどこで起きるかを
      明示的に示している。

■ 動作確認
    Python 3.11.13 (conda env: samurai) / scikit-learn 1.5.1
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# ---------------------------------------------------------------
# 1. 戦略（学習アルゴリズム）の共通インターフェースを定義する
# ---------------------------------------------------------------
class ClassifierStrategy(ABC):
    """分類アルゴリズムの共通インターフェース。"""

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError


# ---------------------------------------------------------------
# 2. 具体的な戦略をそれぞれ実装する
#    内部では scikit-learn の各推定器を使うが、外から見える形は統一する。
# ---------------------------------------------------------------
class LogisticRegressionStrategy(ClassifierStrategy):
    """線形モデル。学習が速く、解釈しやすいのが特徴。"""

    def __init__(self) -> None:
        self._model = LogisticRegression(max_iter=200)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def name(self) -> str:
        return "LogisticRegression"


class RandomForestStrategy(ClassifierStrategy):
    """決定木のアンサンブル。非線形な関係も捉えやすい。"""

    def __init__(self) -> None:
        self._model = RandomForestClassifier(n_estimators=100, random_state=0)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def name(self) -> str:
        return "RandomForest"


class KNeighborsStrategy(ClassifierStrategy):
    """近傍データの多数決で予測するシンプルな手法。"""

    def __init__(self, n_neighbors: int = 5) -> None:
        self._model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def name(self) -> str:
        return f"KNeighbors(k={self._model.n_neighbors})"


# ---------------------------------------------------------------
# 3. 戦略を「利用する側」のクラス（Context）
#    どの戦略が来ても fit()/predict() を呼ぶだけでよく、内部実装に依存しない。
# ---------------------------------------------------------------
class ModelTrainer:
    """渡された ClassifierStrategy を使って学習・評価を行うクラス。"""

    def __init__(self, strategy: ClassifierStrategy) -> None:
        self._strategy = strategy

    def set_strategy(self, strategy: ClassifierStrategy) -> None:
        """実行中にアルゴリズムを入れ替えられる。これがStrategyパターンの肝。"""
        self._strategy = strategy

    def train_and_evaluate(self, X_train, y_train, X_test, y_test) -> float:
        self._strategy.fit(X_train, y_train)
        predictions = self._strategy.predict(X_test)
        accuracy = float(np.mean(predictions == y_test))
        print(f"[{self._strategy.name()}] accuracy={accuracy:.3f}")
        return accuracy


if __name__ == "__main__":
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    trainer = ModelTrainer(strategy=LogisticRegressionStrategy())
    trainer.train_and_evaluate(X_train, y_train, X_test, y_test)

    # 呼び出し側のコード(train_and_evaluate)を変えずに、戦略だけを入れ替えられる
    trainer.set_strategy(RandomForestStrategy())
    trainer.train_and_evaluate(X_train, y_train, X_test, y_test)

    trainer.set_strategy(KNeighborsStrategy(n_neighbors=3))
    trainer.train_and_evaluate(X_train, y_train, X_test, y_test)
