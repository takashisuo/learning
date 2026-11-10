"""
Singleton パターン（生成パターン）
================================

■ 目的
    あるクラスのインスタンスが、アプリケーション全体で「常に1つだけ」
    存在することを保証するパターン。2回目以降のインスタンス化では、
    既存のインスタンスがそのまま返される。

■ いつ使う？
    - 初期化コストが高いオブジェクトを、何度も作り直したくない場合
    - アプリ全体で設定値やリソースを1つに統一したい場合

■ 注意点（重要）
    - Singletonは「グローバル変数」に近く、安易に使うとテストしづらく
      なったり、依存関係が見えにくくなったりする。
    - マルチスレッドで使う場合はロックなどで排他制御が必要（本サンプルは単純化のため省略）。
    - 多くの場合、Pythonでは「モジュールレベルの変数」や設定ライブラリ
      （例: functools.lru_cache を使った関数）で同じ目的を達成できる。

■ 機械学習プロジェクトでの使いどころ
    - 学習済みモデル（pickle/joblibファイル）の読み込みは、ファイルサイズが
      大きいほど時間がかかる。Web API等でリクエストごとに毎回ロードすると
      レスポンスが遅くなるため、アプリ起動時に1度だけロードして共有する。
    - 学習・推論共通で使う前処理設定（スケーラーの平均・分散など）を
      1か所で管理する。

■ 動作確認
    Python 3.11.13 (conda env: samurai) / scikit-learn 1.5.1
"""

from __future__ import annotations

import time
from typing import Optional

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression


class TrainedModelRegistry:
    """学習済みモデルのロードを1回だけ行うSingleton。"""

    _instance: Optional["TrainedModelRegistry"] = None

    def __new__(cls) -> "TrainedModelRegistry":
        if cls._instance is None:
            print("[INFO] 学習済みモデルファイルのロードを実行します（重い処理を想定）...")
            time.sleep(1)  # joblib.load()等の読み込みに時間がかかることを再現
            instance = super().__new__(cls)
            X, y = load_iris(return_X_y=True)
            model = LogisticRegression(max_iter=200)
            model.fit(X, y)  # 本来は学習済みモデルをファイルから読み込むだけの処理
            instance._model = model
            cls._instance = instance
        else:
            print("[INFO] 既存のインスタンスを再利用します（再ロードしない）")
        return cls._instance

    def predict(self, features) -> int:
        return int(self._model.predict([features])[0])


if __name__ == "__main__":
    registry1 = TrainedModelRegistry()  # 1回目: 実際にロード処理が走る
    registry2 = TrainedModelRegistry()  # 2回目: ロードはスキップされ、同じインスタンスを返す

    print("同一インスタンスか？:", registry1 is registry2)
    print("推論結果:", registry1.predict([5.1, 3.5, 1.4, 0.2]))
