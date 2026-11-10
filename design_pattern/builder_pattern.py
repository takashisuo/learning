"""
Builder パターン（生成パターン）
================================

■ 目的
    多くの「任意項目」を持つ複雑なオブジェクトを、メソッドチェーンで
    1ステップずつ組み立てるパターン。コンストラクタの引数が増えすぎて
    読みにくくなる問題（通称: 引数地獄）を解消する。

■ いつ使う？
    - 生成するオブジェクトの構成要素（パラメータ）が多く、しかも
      一部は省略可能（デフォルト値あり）な場合
    - 組み立ての途中経過を読みやすく表現したい場合

■ 機械学習プロジェクトでの使いどころ
    - 前処理（欠損値補完・スケーリング・次元削減）とモデルを組み合わせた
      scikit-learnの `Pipeline` は、組み合わせ次第で構成要素が多くなる。
      Builderを使うことで、「どのステップを入れたか」が一目でわかる
      実験コードになる。

■ 動作確認
    Python 3.11.13 (conda env: samurai) / scikit-learn 1.5.1
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class MLPipelineBuilder:
    """前処理＋モデルのscikit-learn Pipelineを1ステップずつ組み立てるBuilder。

    各メソッドが self を返すことで、メソッドチェーン（.連結）で記述できる。
    """

    def __init__(self) -> None:
        self._steps: List[Tuple[str, BaseEstimator]] = []

    def with_imputer(self, strategy: str = "mean") -> "MLPipelineBuilder":
        """欠損値補完ステップを追加する。"""
        self._steps.append(("imputer", SimpleImputer(strategy=strategy)))
        return self

    def with_scaler(self) -> "MLPipelineBuilder":
        """標準化ステップを追加する。"""
        self._steps.append(("scaler", StandardScaler()))
        return self

    def with_pca(self, n_components: int) -> "MLPipelineBuilder":
        """次元削減ステップを追加する（省略可能）。"""
        self._steps.append(("pca", PCA(n_components=n_components)))
        return self

    def with_classifier(self, estimator: BaseEstimator) -> "MLPipelineBuilder":
        """最終段の分類器を設定する。"""
        self._steps.append(("classifier", estimator))
        return self

    def build(self) -> Pipeline:
        """組み立てた結果を確定させて、scikit-learnのPipelineとして返す。"""
        if not self._steps or self._steps[-1][0] != "classifier":
            raise ValueError("Pipelineの最後はwith_classifier()で分類器を指定してください")
        return Pipeline(self._steps)


def make_iris_with_missing_values(missing_ratio: float = 0.1, random_state: int = 0):
    """欠損値補完(Imputer)の効果を確認するため、意図的に欠損値を混入させる。"""
    X, y = load_iris(return_X_y=True)
    X = X.copy()
    rng = np.random.RandomState(random_state)
    mask = rng.rand(*X.shape) < missing_ratio
    X[mask] = np.nan
    return X, y


if __name__ == "__main__":
    X, y = make_iris_with_missing_values()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    pipeline = (
        MLPipelineBuilder()
        .with_imputer(strategy="mean")
        .with_scaler()
        .with_pca(n_components=2)
        .with_classifier(LogisticRegression(max_iter=200))
        .build()
    )

    print(pipeline)  # 組み立てられたPipelineの構成がそのまま表示される

    pipeline.fit(X_train, y_train)
    accuracy = pipeline.score(X_test, y_test)
    print(f"accuracy={accuracy:.3f}")
