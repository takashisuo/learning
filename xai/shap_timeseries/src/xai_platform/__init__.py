"""xai_platform: 木モデルの予測値を SHAP で「ベースライン + 寄与」に分解する汎用パッケージ。

learning_v2/xai/shap の xai_platform を複製し、入口を ShapAnalyzer(Facade)1つにまとめたもの。
特定のデータセットに依存しない。データ固有の設定(特徴量グループ、シナリオ、単位、ファイル名など)は
すべて呼び出し側から引数で渡す。使い方は README.md を参照。

公開するもの:
    ShapAnalyzer  分析の入口(Facade)
    quick_analyze 1行で一通りの分析(ShapAnalyzer を作って run を呼ぶ)
    Offset        モデルの外で予測値に足す補正項
    Unit          目的変数の単位と表示桁数
    StepResult    メソッドの戻り値(表・図・文章の入れ物。ノートブックでは .show())
    出力の補助    save_csv / save_json / save_figure / analysis_step / produces_table / produces_figure
内部実装は _core/ にあり、直接使う必要はない。
"""
from ._core.io import APP_SOURCE, OutputDirs, StepResult, save_csv, save_json
from ._core.plots import plt, save_figure
from ._core.provenance import MANIFEST, produces_figure, produces_table
from ._core.units import Unit
from ._core.whatif import Scenario
from .analyzer import Offset, ShapAnalyzer, quick_analyze

analysis_step = MANIFEST.step  # with analysis_step("..."): の中で保存したファイルに分析ステップ名を記録する

__all__ = ["ShapAnalyzer", "quick_analyze", "Offset", "Unit", "StepResult", "Scenario", "OutputDirs",
           "save_csv", "save_json", "save_figure", "plt", "analysis_step", "produces_table",
           "produces_figure", "MANIFEST", "APP_SOURCE"]
