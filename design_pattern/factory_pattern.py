"""
Factory パターン（生成パターン）
================================

■ 目的
    「どのクラスのインスタンスを作るか」の判断ロジックを1か所に集め、
    呼び出し側から `if model_type == "random_forest": ... elif ...` のような
    分岐コードを排除するパターン。

■ いつ使う？
    - 実験設定ファイル（YAML/JSON）や環境変数の値（文字列）に応じて、
      生成するモデルを切り替えたい場合
    - 対応モデルが増えても、学習・評価コード側を変更したくない場合

■ 機械学習プロジェクトでの使いどころ
    - 実験管理ツール（MLflow等）の設定で `model_type: random_forest` のように
      文字列でモデル種別を指定し、それに応じたscikit-learn推定器を生成する。
    - ハイパーパラメータ探索で「モデルの種類」自体も探索対象にする場合、
      文字列→インスタンスの変換ロジックを1か所にまとめておくと管理しやすい。

■ 動作確認
    Python 3.11.13 (conda env: samurai) / scikit-learn 1.5.1
"""

from __future__ import annotations

from sklearn.base import ClassifierMixin
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


# ---------------------------------------------------------------
# Factory: 文字列（モデル種別）から scikit-learn の推定器を生成する処理を集約する
# ---------------------------------------------------------------
class ModelFactory:
    """model_type名に応じて適切な分類モデルを生成するFactory。"""

    @staticmethod
    def create(model_type: str, **hyperparams) -> ClassifierMixin:
        if model_type == "logistic_regression":
            return LogisticRegression(max_iter=200, **hyperparams)
        if model_type == "random_forest":
            return RandomForestClassifier(random_state=0, **hyperparams)
        if model_type == "svm":
            return SVC(**hyperparams)
        raise ValueError(f"未対応のmodel_typeです: {model_type}")


if __name__ == "__main__":
    # 実験設定ファイル(config.yaml等)から読み込んだ想定のdict
    experiment_config = {"model_type": "random_forest", "n_estimators": 50, "max_depth": 3}

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    model_type = experiment_config.pop("model_type")
    model = ModelFactory.create(model_type, **experiment_config)

    # 呼び出し側は具体的なクラス名(RandomForestClassifier等)を一切知らなくてよい
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"model={type(model).__name__}, accuracy={accuracy:.3f}")
