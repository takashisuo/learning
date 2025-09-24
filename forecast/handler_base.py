import joblib
import pandas as pd
import logging

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


# 抽象クラスのインポート
from abc import ABC, abstractmethod

class TimeSeriesAbstractHandler(ABC):
    def __init__(self, exog_cols=None):
        self.exog_cols = exog_cols if exog_cols is not None else []
        self._model = None
        self._forecast = None

    @abstractmethod
    def fit(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def predict(self, periods: int, freq: str = "M", future_exog: pd.DataFrame = None) -> pd.DataFrame:
        pass

    @abstractmethod
    def evaluate(self, df: pd.DataFrame) -> dict:
        pass

    @abstractmethod
    def plot(self, save_path: str = "forecast_plot.png"):
        pass
    
    def save_model(self, path="forecast_model.pkl"):
        """ モデル保存 """
        joblib.dump(self._model, path)

    def load_model(self, path="forecaset_model.pkl"):
        """ モデルロード """
        self._model = joblib.load(path)