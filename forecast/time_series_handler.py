from prophet import Prophet
import numpy as np
import joblib
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from pmdarima.model_selection import train_test_split

import matplotlib.pyplot as plt
# 抽象クラスのインポート
from abc import ABC, abstractmethod

class TimeSeriesAbstractHandler(ABC):
    def __init__(self, exog_cols=None):
        self.exog_cols = exog_cols if exog_cols is not None else []
        self.model = None
        self.forecast = None

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

    def transform_to_prophet_format(self,
                                    df,
                                    date_col: str = 'date',
                                    value_col: str = 'value',
                                    exog_cols=None
                                    ) -> pd.DataFrame:
        """
        任意のDataFrameをProphet用(ds/y+外生変数列)の形式へ変換

        Parameters:
            df (pd.DataFrame): 入力DataFrame
            date_col (str): 日時のカラム名
            value_col (str): 目的変数のカラム名
            exog_cols (list or None): 外生変数として使うカラム名リスト

        Returns:
            pd.DataFrame: Prophet用の'ds', 'y', (exog_cols...)列のDataFrame
        """
        print(f"Transforming DataFrame to Prophet format with date_col='{date_col}', value_col='{value_col}', exog_cols={exog_cols}")
        print(f"Original DataFrame: \n{df.head()}")
        use_cols = [date_col, value_col]
        if exog_cols is not None:
            use_cols += exog_cols
        df_prophet = df[use_cols].copy()
        df_prophet.rename(columns={date_col: 'ds', value_col: 'y'}, inplace=True)
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
        print(f"Transformed DataFrame: \n{df_prophet.head()}")
        return df_prophet
    
    def save_model(self, path="forecast_model.pkl"):
        """ モデル保存 """
        joblib.dump(self._model, path)

    def load_model(self, path="forecaset_model.pkl"):
        """ モデルロード """
        self._model = joblib.load(path)


class ProphetTrainingHandler(TimeSeriesAbstractHandler):
    def __init__(self, exog_cols=None):
        """ Initializes the Prophet model and necessary attributes.
        Prophet requires a DataFrame with two columns: 'ds' for the date and 'y' for the value to be forecasted.
        'ds' must be of a date type (e.g., datetime), and 'y' must be numeric.
        'eog_cols' are optional exogenous variables to include in the model.
        """
        super().__init__(exog_cols)
        self._model = Prophet()
        for col in self.exog_cols:
            self._model.add_regressor(col)
        self._train = None
        self._test = None
        self._train_pred = None
        self._test_pred = None

    def fit(self, df: pd.DataFrame):
        # df should have columns 'ds' (date) and 'y' (value)
        if 'ds' not in df.columns or 'y' not in df.columns:
            raise ValueError("DataFrame must contain 'ds' and 'y' columns.")

        self._model.fit(df)

    def predict(self, periods: int, freq:str = 'M', future_exog: pd.DataFrame = None) -> pd.DataFrame:
        # periods is the number of future periods to predict
        # freq is the frequency of the periods (e.g., 'D' for daily, 'M' for monthly)
        # forecast is a DataFrame with the predictions, including 'ds' and 'yhat' columns

        # return: forecast DataFrame: columns: 'ds', 'yhat', 'yhat_lower', 'yhat_upper'
        future = self._model.make_future_dataframe(periods=periods, freq=freq)

        # 外生変数がある場合は future_exog を連結
        if self.exog_cols:
            if future_exog is None:
                raise ValueError("Exogenous variables are required but future_exog is None.")
            if not set(self.exog_cols).issubset(future_exog.columns):
                raise ValueError(f"future_exog must contain columns: {self.exog_cols}")
            future = future.merge(future_exog, on="ds", how="left")
        
        self.forecast = self._model.predict(future)
        return self.forecast

    def evaluate(
            self,
            df: pd.DataFrame,
            test_size: int = 12,
            freq: str = "M",
            call_plot: bool = False) -> dict:
        # df should have columns 'ds' (date) and 'y' (value)
        # return: dict with keys: 'MSE', 'MAE', 'MAPE'
        # if called fit and predict, set df to 
        if 'ds' not in df.columns or 'y' not in df.columns:
            raise ValueError("DataFrame must contain 'ds' and 'y' columns.")
        
        train, test = train_test_split(df, test_size=test_size)
        self._train = train
        self._test = test

        self.fit(train)
        test_forecast: pd.DataFrame = self.predict(
            periods=test_size,
            freq=freq,
            future_exog=test[self.exog_cols] if self.exog_cols else None
            )

        test_true = test['y'].values
        self._train_pred = test_forecast['yhat'][:len(train)].values
        self._test_pred = test_forecast['yhat'][-len(test):].values

        mse = mean_squared_error(test_true, self._test_pred)
        mae = mean_absolute_error(test_true, self._test_pred)
        mape = mean_absolute_percentage_error(test_true, self._test_pred)

        if call_plot:
            self.plot()

        return {
            'MSE': mse,
            'MAE': mae,
            'MAPE': mape
        }

    def plot(self, save_path: str = "prophet_forecast_train_test.png"):
        """ 予測結果の可視化. evaluate 実行後に使用することを想定.
        Parameters:
            save_path (str): 画像保存パス
        """

        fig, ax = plt.subplots(figsize=(10, 6))

        # describe the training data
        if self._train is not None:
            ax.plot(self._train['ds'], self._train['y'], label='actural(Train)', color='blue')

        # describe the test data
        if self._test is not None:
            ax.plot(self._test['ds'], self._test['y'], label='actural(Test)', color='orange')

        # describe the training predictions
        if self._train_pred is not None and self._train is not None:
            ax.plot(self._train['ds'], self._train_pred, label='predicted(Train)', color='cyan', linestyle='--')

        # describe the test predictions
        if self._test_pred is not None and self._test is not None:
            ax.plot(self._test['ds'], self._test_pred, label='predicted(Test)', color='red', linestyle='--')

        ax.legend()
        ax.set_title('Prophet Forecast')

        # save the figure
        plt.tight_layout()
        plt.savefig(save_path)

        plt.show()

    def get_train_data(self) -> pd.DataFrame:
        return self._train

class ProphetProductHandler(TimeSeriesAbstractHandler):
    def __init__(self, exog_cols=None):
        """ Initializes the Prophet model and necessary attributes.
        Prophet requires a DataFrame with two columns: 'ds' for the date and 'y' for the value to be forecasted.
        """
        super().__init__(exog_cols)
        self._model = Prophet()
        for col in self.exog_cols:
            self._model.add_regressor(col)
        self._forecast = None
        self._train = None

    def fit(self, df: pd.DataFrame):
        # df should have columns 'ds' (date) and 'y' (value)
        if 'ds' not in df.columns or 'y' not in df.columns:
            raise ValueError("DataFrame must contain 'ds' and 'y' columns.")

        self._model.fit(df)
        self._train = df.copy()

    def predict(self, periods: int, freq:str = 'M', future_exog: pd.DataFrame = None) -> pd.DataFrame:
        # periods is the number of future periods to predict
        # freq is the frequency of the periods (e.g., 'D' for daily, 'M' for monthly)
        # forecast is a DataFrame with the predictions, including 'ds' and 'yhat' columns

        # return: forecast DataFrame: columns: 'ds', 'yhat', 'yhat_lower', 'yhat_upper'
        future = self._model.make_future_dataframe(periods=periods, freq=freq)

        # 外生変数がある場合は future_exog を連結
        if self.exog_cols:
            if future_exog is None:
                raise ValueError("Exogenous variables are required but future_exog is None.")
            if not set(self.exog_cols).issubset(future_exog.columns):
                raise ValueError(f"future_exog must contain columns: {self.exog_cols}")
            future = future.merge(future_exog, on="ds", how="left")
        
        self._forecast = self._model.predict(future)
        return self._forecast

    def evaluate(self, df: pd.DataFrame) -> dict:
        raise NotImplementedError("Evaluation is not implemented for production handler.")

    def plot(self, save_path: str = "prophet_product_forecast.png"):
        """ 予測結果の可視化.
        Parameters:
            save_path (str): 画像保存パス
        """
        if self._forecast is None:
            raise ValueError("No forecast to plot.")
        forecast_future = self._forecast[self._forecast['ds'] > self._train['ds'].max()]

        fig, ax = plt.subplots(figsize=(10, 6))
        # 訓練データの実績値を表示
        ax.plot(self._train['ds'], self._train['y'], label='actural(Train)', color='blue')
        # 予測値
        ax.plot(forecast_future['ds'], forecast_future['yhat'], label='forecast', color='red')
        # 予測区間
        ax.fill_between(forecast_future['ds'],
                        forecast_future['yhat_lower'],
                        forecast_future['yhat_upper'],
                        color='pink', alpha=0.3)
        ax.legend()
        ax.set_title('Prophet Product Forecast')

        # save the figure
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()

    def set_train_data(self, train_df: pd.DataFrame):
        self._train = train_df.copy()
