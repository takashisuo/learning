"""Titanic生存予測を題材にした、MLflowロギングの一連の流れを示すサンプルスクリプト。

実行方法:
    (Windows/Linux共通)
    export MLFLOW_TRACKING_URI=http://<サーバーIP>:5000   # Windowsは set コマンド
    python src/train.py

前提:
    - start_mlflow_server.bat 等でMLflow serverが起動していること。
    - MLFLOW_TRACKING_URI が未設定の場合は http://localhost:5000 にフォールバックする
      (mlflow_utils.get_tracking_uri を参照)。

実行後の確認ポイント:
    - MLflow UI (http://<サーバーIP>:5000) で今回のrunが記録されていること。
    - runのartifactsに model/ と confusion_matrix.png が保存されていること。
    - run詳細画面のartifact_locationが `mlflow-artifacts:/...` になっていること
      (`file://...` になっている場合はサーバー起動設定のミス)。
    - 別OS(Windows/Linux)のマシンからも、同じ MLFLOW_TRACKING_URI を設定した上で
      `mlflow.artifacts.download_artifacts(run_id=..., artifact_path="model")` 等を
      実行し、同じrunのartifactを取得できること。
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # GUI不要のバックエンド(サーバー/CI環境でも動作させるため)
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from data_loader import load_titanic_data, split_features_target
from mlflow_utils import run_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "titanic-survival-classification"

MODEL_PARAMS = {
    "model_name": "RandomForestClassifier",
    "n_estimators": 100,
    "max_depth": 5,
    "random_state": 42,
}


def _create_confusion_matrix_artifact(y_true, y_pred, output_dir: Path) -> Path:
    """混同行列の画像を作成し、ファイルとして保存する。

    保存先は呼び出し元が指定した一時ディレクトリ配下(相対的に解決されるパス)であり、
    OS依存の絶対パスをハードコードしない。
    """
    cm = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["not survived", "survived"])

    fig, ax = plt.subplots(figsize=(5, 5))
    display.plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix")

    output_path = output_dir / "confusion_matrix.png"
    fig.savefig(output_path)
    plt.close(fig)

    return output_path


def main() -> None:
    logger.info("Titanicデータセットを読み込みます。")
    df = load_titanic_data()
    x, y = split_features_target(df)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=MODEL_PARAMS["random_state"]
    )

    logger.info("モデルを学習します: %s", MODEL_PARAMS)
    model = RandomForestClassifier(
        n_estimators=MODEL_PARAMS["n_estimators"],
        max_depth=MODEL_PARAMS["max_depth"],
        random_state=MODEL_PARAMS["random_state"],
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }
    logger.info("評価結果: %s", metrics)

    # 一時ディレクトリに混同行列画像を作成し、artifactとしてMLflowへアップロードする。
    # (proxied artifact accessにより、実ファイルはサーバー側へHTTP経由で送られる)
    with tempfile.TemporaryDirectory() as tmp_dir:
        cm_path = _create_confusion_matrix_artifact(y_test, y_pred, Path(tmp_dir))

        run_id = run_experiment(
            experiment_name=EXPERIMENT_NAME,
            params=MODEL_PARAMS,
            metrics=metrics,
            model=model,
            model_flavor="sklearn",
            artifacts=[cm_path],
            run_name="rf-baseline",
        )

    logger.info("MLflowへの記録が完了しました。run_id=%s", run_id)
    logger.info(
        "MLflow UIで確認してください: <MLFLOW_TRACKING_URIのURL>/#/experiments "
        "(experiment名: %s)",
        EXPERIMENT_NAME,
    )


if __name__ == "__main__":
    main()
