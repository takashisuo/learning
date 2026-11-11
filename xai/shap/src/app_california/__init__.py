"""app_california: California Housing を題材にしたアプリケーション側。

データ固有の設定は config.py に集約し、分析の各ステップは pipeline.py が
xai_platform の I/F を呼び出して実行する(CSV 作成・描画も platform 側の関数を使う)。
"""
