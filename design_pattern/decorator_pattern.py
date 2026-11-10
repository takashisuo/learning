"""
Decorator パターン（構造パターン）
================================

■ 目的
    既存のクラス（オブジェクト）に対して、機能を「後付け」で追加するパターン。
    継承を使わずに、複数の機能を自由に組み合わせて積み重ねることができる。

    ※ Python の `@デコレータ` 構文（関数デコレータ）と発想は同じだが、
      ここでは GoF（デザインパターンの元祖本）の「クラスでラップする」
      実装方法を扱う。

■ いつ使う？
    - 「ログを出す」「リトライする」「キャッシュする」など、本質的な処理とは
      別軸の機能（横断的関心事）を、本体のクラスを汚さずに追加したい場合
    - 機能の組み合わせパターンが多く、継承で全部のクラスを作ると爆発する場合

■ 機械学習プロジェクトでの使いどころ
    - 学習データを共有ストレージ（NFS/S3等）から取得する処理は、
      ネットワークが不安定だと「まれに失敗する」ことがある。
    - 同じ実験設定（ハイパーパラメータ）で何度も再学習するのはコストが高いため、
      結果をキャッシュして再利用したい。
    - 「どの実験設定で、どの結果が出たか」を必ずログに残したい。
      → これらを本体の学習処理(SimpleTrainer)を汚さずに、後付けで重ねがけする。

■ 動作確認
    Python 3.11.13 (conda env: samurai) / scikit-learn 1.5.1
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------
# 1. 元になるインターフェースと、本体の実装（飾り付けされる側）
# ---------------------------------------------------------------
class TrainerComponent(ABC):
    @abstractmethod
    def run(self, experiment_id: str) -> float:
        """指定された実験設定(experiment_id)で学習・評価し、accuracyを返す。"""
        raise NotImplementedError


class SimpleTrainer(TrainerComponent):
    """実験設定ごとにハイパーパラメータを切り替えて学習する本体クラス。

    共有ストレージから学習データを取得する処理を想定し、
    30%の確率で接続エラーを再現する（Retryデコレータの動作確認用）。
    """

    EXPERIMENTS: Dict[str, Dict] = {
        "baseline": {"max_iter": 100},
        "tuned": {"max_iter": 500, "C": 0.5},
    }

    def __init__(self, X_train, y_train, X_test, y_test) -> None:
        self._X_train, self._y_train = X_train, y_train
        self._X_test, self._y_test = X_test, y_test

    def run(self, experiment_id: str) -> float:
        if random.random() < 0.3:
            raise ConnectionError("学習データの取得に失敗しました（共有ストレージへの接続エラーを再現）")

        params = self.EXPERIMENTS[experiment_id]
        model = LogisticRegression(**params)
        model.fit(self._X_train, self._y_train)
        return float(model.score(self._X_test, self._y_test))


# ---------------------------------------------------------------
# 2. Decoratorの基底クラス
#    TrainerComponentを実装しつつ、内部に「ラップ対象」を保持する
# ---------------------------------------------------------------
class TrainerDecorator(TrainerComponent):
    def __init__(self, wrapped: TrainerComponent) -> None:
        self._wrapped = wrapped

    def run(self, experiment_id: str) -> float:
        return self._wrapped.run(experiment_id)


# ---------------------------------------------------------------
# 3. 具体的なDecoratorたち
# ---------------------------------------------------------------
class LoggingDecorator(TrainerDecorator):
    """実験ID・結果をログ（標準出力）に記録する。"""

    def run(self, experiment_id: str) -> float:
        print(f"[LOG] 実験開始: experiment_id={experiment_id!r}")
        accuracy = super().run(experiment_id)
        print(f"[LOG] 実験完了: experiment_id={experiment_id!r}, accuracy={accuracy:.3f}")
        return accuracy


class RetryDecorator(TrainerDecorator):
    """データ取得失敗時に指定回数まで再試行する。"""

    def __init__(self, wrapped: TrainerComponent, max_retries: int = 3) -> None:
        super().__init__(wrapped)
        self._max_retries = max_retries

    def run(self, experiment_id: str) -> float:
        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                return super().run(experiment_id)
            except ConnectionError as exc:
                last_error = exc
                print(f"[RETRY] {attempt}回目の実行に失敗: {exc}")
        raise RuntimeError(f"{self._max_retries}回再試行しましたが失敗しました") from last_error


class CachingDecorator(TrainerDecorator):
    """同じexperiment_idの結果をメモリにキャッシュし、再学習コストを節約する。"""

    def __init__(self, wrapped: TrainerComponent) -> None:
        super().__init__(wrapped)
        self._cache: Dict[str, float] = {}

    def run(self, experiment_id: str) -> float:
        if experiment_id in self._cache:
            print(f"[CACHE] キャッシュから返却: experiment_id={experiment_id!r}")
            return self._cache[experiment_id]
        accuracy = super().run(experiment_id)
        self._cache[experiment_id] = accuracy
        return accuracy


if __name__ == "__main__":
    random.seed(0)  # 結果を再現可能にする（30%失敗の再現用）

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # 内側から「リトライ → ログ → キャッシュ」の順に重ねがけする
    trainer: TrainerComponent = CachingDecorator(
        LoggingDecorator(RetryDecorator(SimpleTrainer(X_train, y_train, X_test, y_test)))
    )

    print("--- 1回目の実行 ---")
    trainer.run("tuned")

    print("--- 2回目の実行（同じexperiment_id → キャッシュが効くはず） ---")
    trainer.run("tuned")
