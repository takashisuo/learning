"""
Template Method パターン（振る舞いパターン）
================================

■ 目的
    処理の「全体の流れ（骨格）」は親クラスで固定しつつ、
    一部のステップだけをサブクラスでカスタマイズできるようにするパターン。
    「流れは統一したいが、一部だけ差し替えたい」場合に最適。

■ いつ使う？
    - 複数の実験が似たステップ構成を持つが、一部のステップだけ実装が異なる場合
    - 処理順序を呼び出し側やサブクラスに自由に変えさせたくない場合
      （データ漏洩(data leakage)などの順序ミスによるバグを防げる）

■ 機械学習プロジェクトでの使いどころ
    - 機械学習の実験は多くの場合
        1. データ読み込み (load_data)
        2. 前処理（欠損値処理・スケーリング等） (preprocess)
        3. モデルの学習 (train)
        4. 評価 (evaluate)
        5. レポート出力 (report)
      という共通の流れを持つ。この「流れ」を親クラスに固定することで、
      「前処理だけ変えたい」「モデルだけ変えたい」といった実験を、
      流れを壊さずに安全に試せる。

■ 動作確認
    Python 3.11.13 (conda env: samurai) / scikit-learn 1.5.1
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class ExperimentPipeline(ABC):
    """機械学習実験の「型」を定義する基底クラス。"""

    def run(self) -> float:
        """テンプレートメソッド: 処理の流れ（順序）はここで固定する。

        サブクラスはこのメソッド自体を上書きしない。各ステップだけを実装する。
        """
        X_train, X_test, y_train, y_test = self.load_data()
        X_train, X_test = self.preprocess(X_train, X_test)
        model = self.train(X_train, y_train)
        accuracy = self.evaluate(model, X_test, y_test)
        self.report(accuracy)
        return accuracy

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """データ読み込み処理。デフォルト実装を提供し、必要ならサブクラスで上書き可能。"""
        X, y = load_iris(return_X_y=True)
        return train_test_split(X, y, test_size=0.3, random_state=0)

    def preprocess(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """前処理。デフォルトは何もしない（そのまま返す）。サブクラスで上書き可能。"""
        return X_train, X_test

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """モデルの学習処理。サブクラスで実装必須。"""
        raise NotImplementedError

    def evaluate(self, model, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """評価処理。デフォルトはaccuracyを計算する。サブクラスで上書き可能。"""
        return float(model.score(X_test, y_test))

    def report(self, accuracy: float) -> None:
        """レポート出力。デフォルトは標準出力に表示するだけ。サブクラスで上書き可能。"""
        print(f"[{type(self).__name__}] accuracy={accuracy:.3f}")


# ---------------------------------------------------------------
# 具体的な実験: 前処理なしのロジスティック回帰
# ---------------------------------------------------------------
class PlainLogisticRegressionExperiment(ExperimentPipeline):
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)
        return model


# ---------------------------------------------------------------
# 別バリエーション: preprocessだけを変えた例（流れ自体は変えない）
# ---------------------------------------------------------------
class ScaledLogisticRegressionExperiment(PlainLogisticRegressionExperiment):
    """特徴量を標準化(StandardScaler)してから学習する実験。"""

    def preprocess(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)  # テストデータは学習時の統計量で変換する（リークを防ぐ）
        return X_train_scaled, X_test_scaled


if __name__ == "__main__":
    plain_experiment = PlainLogisticRegressionExperiment()
    plain_experiment.run()

    scaled_experiment = ScaledLogisticRegressionExperiment()
    scaled_experiment.run()
