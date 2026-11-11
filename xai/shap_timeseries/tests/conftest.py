import pytest

from src.pipeline import BikeAnalysis


@pytest.fixture(scope="session")
def analysis():
    """説明用モデル(LightGBM[level])を学習し SHAP を計算済みの分析オブジェクト(ファイル出力なし)。"""
    a = BikeAnalysis()
    a._explain()
    return a
