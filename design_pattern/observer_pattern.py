"""
Observer パターン（振る舞いパターン）
================================

■ 目的
    あるオブジェクト（Subject/発行者）の状態変化を、登録された複数の
    オブジェクト（Observer/購読者）へ自動的に通知するパターン。
    発行者は「誰が見ているか」を知らなくてよく、購読者の追加・削除も自由にできる。

■ いつ使う？
    - 1つのイベントに対して、複数の異なる処理（ログ出力、履歴保存、早期終了判定など）
      を反応させたい場合
    - 発行側と購読側を疎結合（依存しない関係）に保ちたい場合

■ 機械学習プロジェクトでの使いどころ
    - ニューラルネットワーク等の反復学習では、1エポックごとに
      「ログ出力」「学習曲線用の履歴保存」「Early Stopping判定」など
      複数の処理を反応させたい。これはKeras/PyTorch Lightningの
      `callbacks` 機構そのものであり、その正体はObserverパターンである。
    - 新しい監視処理（例: Slack通知）を追加しても、学習ループ本体
      （Trainerクラス）のコードは変更不要になる。

■ 動作確認
    Python 3.11.13 (conda env: samurai) / scikit-learn 1.5.1
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------
# 1. Observer（購読者）の共通インターフェース
# ---------------------------------------------------------------
class TrainingObserver(ABC):
    @abstractmethod
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        """1エポックの学習が終わったときの処理。"""
        raise NotImplementedError

    def on_train_end(self) -> None:
        """学習が完全に終わったときの処理（任意でオーバーライド）。"""


# ---------------------------------------------------------------
# 2. Subject（発行者）: 1エポックごとに購読者全員に通知する学習ループ
# ---------------------------------------------------------------
class Trainer:
    def __init__(self) -> None:
        self._observers: List[TrainingObserver] = []

    def subscribe(self, observer: TrainingObserver) -> None:
        self._observers.append(observer)

    def unsubscribe(self, observer: TrainingObserver) -> None:
        self._observers.remove(observer)

    def fit(self, X_train, y_train, X_val, y_val, classes, max_epochs: int = 50) -> None:
        """SGDClassifierをpartial_fitで1エポックずつ進め、毎エポック観測者に通知する。"""
        model = SGDClassifier(loss="log_loss", random_state=0)

        for epoch in range(1, max_epochs + 1):
            model.partial_fit(X_train, y_train, classes=classes)
            val_accuracy = float(model.score(X_val, y_val))
            metrics = {"epoch": epoch, "val_accuracy": val_accuracy}

            for observer in self._observers:
                observer.on_epoch_end(epoch, metrics)

            # EarlyStoppingObserverが「打ち切り」を要求したら学習ループを止める
            if any(isinstance(o, EarlyStoppingObserver) and o.should_stop for o in self._observers):
                print(f"[Trainer] epoch={epoch} でEarly Stoppingが発動したため学習を終了します")
                break

        for observer in self._observers:
            observer.on_train_end()


# ---------------------------------------------------------------
# 3. 具体的なObserverたち
# ---------------------------------------------------------------
class ConsoleLoggerObserver(TrainingObserver):
    """毎エポックの進捗を標準出力に表示する。"""

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        print(f"[epoch {epoch}] val_accuracy={metrics['val_accuracy']:.3f}")


class MetricsHistoryObserver(TrainingObserver):
    """学習曲線の描画などに使うため、エポックごとの指標を履歴として保存する。"""

    def __init__(self) -> None:
        self.history: List[Dict[str, float]] = []

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        self.history.append(metrics)

    def on_train_end(self) -> None:
        print(f"[MetricsHistory] {len(self.history)}エポック分の履歴を記録しました")


class EarlyStoppingObserver(TrainingObserver):
    """精度の改善が一定エポック数(patience)見られない場合、学習打ち切りを要求する。"""

    def __init__(self, patience: int = 3) -> None:
        self._patience = patience
        self._best_accuracy = -1.0
        self._epochs_without_improvement = 0
        self.should_stop = False

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        if metrics["val_accuracy"] > self._best_accuracy:
            self._best_accuracy = metrics["val_accuracy"]
            self._epochs_without_improvement = 0
        else:
            self._epochs_without_improvement += 1

        if self._epochs_without_improvement >= self._patience:
            self.should_stop = True


if __name__ == "__main__":
    X, y = load_iris(return_X_y=True)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=0)

    trainer = Trainer()
    history_observer = MetricsHistoryObserver()

    trainer.subscribe(ConsoleLoggerObserver())
    trainer.subscribe(history_observer)
    trainer.subscribe(EarlyStoppingObserver(patience=3))

    trainer.fit(X_train, y_train, X_val, y_val, classes=np.unique(y), max_epochs=50)
