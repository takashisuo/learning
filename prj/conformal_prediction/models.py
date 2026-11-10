"""予測に使うベースモデル。"""

from __future__ import annotations

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression


class TrendResidualRegressor(BaseEstimator, RegressorMixin):
    """線形トレンド成分 + 残差成分(任意の回帰モデル)を組み合わせた回帰モデル。

    GBM 等の決定木ベースのモデルは学習データの範囲外(未来)への外挿が苦手で、
    右肩上がり/下がりのトレンドを持つ時系列では将来予測が頭打ちになりやすい。
    そこでトレンドは外挿が得意な線形回帰で分離して求め、木モデルには
    トレンド除去後の残差(ラグ特徴量などで説明しやすい変動)だけを学習させる。

    BaseEstimator/RegressorMixin を継承しているため sklearn.base.clone() が
    でき、MAPIE のようにモデルを内部で複製して使うライブラリにもそのまま渡せる。
    """

    def __init__(self, residual_model, trend_col: str = "day_index"):
        self.residual_model = residual_model
        self.trend_col = trend_col
        self.trend_model = LinearRegression()

    def fit(self, X, y) -> "TrendResidualRegressor":
        trend_X = X[[self.trend_col]]
        self.trend_model.fit(trend_X, y)
        residual_y = y - self.trend_model.predict(trend_X)
        self.residual_model.fit(X, residual_y)
        return self

    def predict(self, X):
        trend_X = X[[self.trend_col]]
        return self.trend_model.predict(trend_X) + self.residual_model.predict(X)
