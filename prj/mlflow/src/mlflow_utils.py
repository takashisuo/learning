"""MLflow関連処理の薄いラッパー。

train.py側がMLflowのAPI詳細(tracking_uriの設定方法、flavorごとのlog_model呼び分け等)
を意識しなくてよいようにするための最小限の抽象化。

このモジュール以外の場所で mlflow_env/ の実体パスを直接参照しないこと。
クライアントは常に MLFLOW_TRACKING_URI 経由(HTTP)でサーバーとやり取りする。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import mlflow

logger = logging.getLogger(__name__)

DEFAULT_TRACKING_URI = "http://localhost:5000"

# model_flavor文字列 -> 対応する mlflow.<flavor> モジュールのlog_model呼び出し
_SUPPORTED_FLAVORS = {"sklearn"}


def get_tracking_uri() -> str:
    """環境変数 MLFLOW_TRACKING_URI からtracking_uriを取得する。

    未設定の場合はデフォルト値にフォールバックし、警告ログを出す。
    IPアドレスやポートをコード中にハードコードしないためのルールを守るための関数。
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        logger.warning(
            "環境変数 MLFLOW_TRACKING_URI が未設定のため、デフォルト値 %s を使用します。"
            "実際のサーバーに接続する場合は "
            "`set MLFLOW_TRACKING_URI=http://<サーバーIP>:5000` (Windows) / "
            "`export MLFLOW_TRACKING_URI=http://<サーバーIP>:5000` (Linux) を設定してください。",
            DEFAULT_TRACKING_URI,
        )
        tracking_uri = DEFAULT_TRACKING_URI
    return tracking_uri


def setup_experiment(experiment_name: str) -> None:
    """tracking_uriの設定 + experimentの設定をまとめて行う。"""
    tracking_uri = get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info(
        "MLflow設定完了: tracking_uri=%s, experiment=%s", tracking_uri, experiment_name
    )


def _log_model(model: Any, model_flavor: str, artifact_path: str) -> None:
    """model_flavorに応じて対応する mlflow.<flavor>.log_model を呼び出す。

    過度な汎用化はせず、サポートするflavorを明示的に列挙する。
    """
    if model_flavor == "sklearn":
        mlflow.sklearn.log_model(model, name=artifact_path)
    else:
        raise ValueError(
            f"未対応のmodel_flavorです: {model_flavor!r}. "
            f"対応済み: {sorted(_SUPPORTED_FLAVORS)}"
        )


def run_experiment(
    experiment_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    model: Any,
    model_flavor: str = "sklearn",
    artifacts: list[str | Path] | None = None,
    run_name: str | None = None,
) -> str:
    """1回分の学習結果(パラメータ・メトリクス・モデル・任意artifact)をMLflowに記録する。

    Args:
        experiment_name: experiment名。
        params: mlflow.log_param(s) に渡すパラメータ群。
        metrics: mlflow.log_metric(s) に渡すメトリクス群。
        model: 学習済みモデルオブジェクト。
        model_flavor: モデルの種類("sklearn"など)。内部でflavorに応じた
            log_model関数を呼び分けるための識別子。
        artifacts: 追加で保存したいファイルパスのリスト(混同行列の画像など)。
            指定したファイルは mlflow.log_artifact() 経由でproxied artifact storeへ
            アップロードされる(ローカルファイルパスがそのままDBに記録されるわけではない)。
        run_name: runの表示名(省略可)。

    Returns:
        実行されたrunのrun_id。
    """
    setup_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        _log_model(model, model_flavor, artifact_path="model")

        if artifacts:
            for artifact_path in artifacts:
                mlflow.log_artifact(str(artifact_path))

        run_id = run.info.run_id
        artifact_uri = mlflow.get_artifact_uri()
        logger.info("run_id=%s artifact_uri=%s", run_id, artifact_uri)

        if not artifact_uri.startswith("mlflow-artifacts:"):
            logger.warning(
                "artifact_uriが 'mlflow-artifacts:' で始まっていません(実際: %s)。"
                "サーバーが --artifacts-destination で proxied access 構成になっているか"
                "確認してください。",
                artifact_uri,
            )

    return run_id
