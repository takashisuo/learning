# Prophet, NeuralProphet, statsmodels, statsforecast などのライブラリを使用した予測を行うためのクラス
# 外生変数のサポートも含む
# 抽象クラス(ABC)を定義し、各ライブラリごとに具体的な実装を行う
from abc import ABC, abstractmethod

class ForecastProxy(ABC):
    @abstractmethod
    def fit(self, Y_df, S_df, Exog_df=None):
        """モデルの学習を行う"""
        pass

    @abstractmethod
    def predict(self, future_df, Exog_df=None):
        """予測を行う"""
        pass

    @abstractmethod
    def evaluate(self, Y_df, S_df, Exog_df=None):
        """モデルの評価を行う"""
        pass


class ProphetForecastProxy(ForecastProxy):
    def __init__(self):
        # Prophetモデルの初期化
        pass

    def fit(self, Y_df, S_df, Exog_df=None):
        # Prophetモデルの学習を実装
        pass

    def predict(self, future_df, Exog_df=None):
        # Prophetモデルの予測を実装
        pass

    def evaluate(self, Y_df, S_df, Exog_df=None):
        # Prophetモデルの評価を実装
        pass


class NeuralProphetForecastProxy(ForecastProxy):
    def __init__(self):
        # NeuralProphetモデルの初期化
        pass

    def fit(self, Y_df, S_df, Exog_df=None):
        # NeuralProphetモデルの学習を実装
        pass

    def predict(self, future_df, Exog_df=None):
        # NeuralProphetモデルの予測を実装
        pass

    def evaluate(self, Y_df, S_df, Exog_df=None):
        # NeuralProphetモデルの評価を実装
        pass


class StatsForecastProxy(ForecastProxy):
    def __init__(self):
        # statsforecastモデルの初期化
        pass

    def fit(self, Y_df, S_df, Exog_df=None):
        # statsforecastモデルの学習を実装
        pass

    def predict(self, future_df, Exog_df=None):
        # statsforecastモデルの予測を実装
        pass

    def evaluate(self, Y_df, S_df, Exog_df=None):
        # statsforecastモデルの評価を実装
        pass