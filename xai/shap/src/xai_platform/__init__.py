"""xai_platform: 回帰モデルの予測値を SHAP で分解するための汎用 I/F 群。

特定のデータセットに依存しない。データ固有の設定(特徴量グループ、シナリオ、単位、
ファイル名など)はすべて呼び出し側(アプリケーション)から引数で渡す。

モジュール:
    units    目的変数の単位と表記(Unit)
    io       出力先ディレクトリと CSV/JSON 保存、ノートブック表示用の StepResult
    data     Dataset と train/test 分割
    model    学習・評価(回帰指標)
    explain  SHAP 計算・加法性チェック・分解表・相互作用・グループ集計
    whatif   再予測による What-if と ΔSHAP 分解
    plots    作図(日本語フォント設定含む)
"""
from .units import Unit

__all__ = ["Unit"]
