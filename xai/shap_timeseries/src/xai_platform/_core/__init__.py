"""xai_platform の内部実装。利用者は xai_platform.ShapAnalyzer を使い、ここを直接呼ばない。

units      目的変数の単位と表記
io         出力先ディレクトリ・CSV/JSON 保存・StepResult
provenance 出力ファイルの来歴記録(@produces_table / @produces_figure / MANIFEST)
explain    SHAP 計算・加法性・線形モデル検証・依存表・相互作用
decompose  グループ単位の分解・要約文・2行の差・区分ごとの平均
whatif     再予測による What-if と ΔSHAP(1行 / 複数行)
plots      作図(日本語フォント設定含む)
"""
