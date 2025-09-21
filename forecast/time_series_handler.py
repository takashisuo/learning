from prophet import Prophet
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from pmdarima.model_selection import train_test_split

import matplotlib.pyplot as plt
# 抽象クラスのインポート
from abc import ABC, abstractmethod

class TimeSeriesAbstractHandler(ABC):
    @abstractmethod
    def fit(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def predict(self, periods: int) -> pd.DataFrame:
        pass

    @abstractmethod
    def evaluate(self, df: pd.DataFrame) -> dict:
        pass

    @abstractmethod
    def plot(self, df: pd.DataFrame = None):
        pass

class ProphetHandler(TimeSeriesAbstractHandler):
    def __init__(self):
        """ Initializes the Prophet model and necessary attributes.
        Prophet requires a DataFrame with two columns: 'ds' for the date and 'y' for the value to be forecasted.
        'ds' must be of a date type (e.g., datetime), and 'y' must be numeric.
        """
        self.model = Prophet()
        self.forecast = None
        self.train = None
        self.test = None
        self.train_pred = None
        self.test_pred = None

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

    def fit(self, df: pd.DataFrame):
        # df should have columns 'ds' (date) and 'y' (value)
        if 'ds' not in df.columns or 'y' not in df.columns:
            raise ValueError("DataFrame must contain 'ds' and 'y' columns.")

        self.model.fit(df)

    def predict(self, periods: int, freq:str = 'M') -> pd.DataFrame:
        # periods is the number of future periods to predict
        # freq is the frequency of the periods (e.g., 'D' for daily, 'M' for monthly)
        # forecast is a DataFrame with the predictions, including 'ds' and 'yhat' columns
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        self.forecast = self.model.predict(future)
        return self.forecast

    def evaluate(self, df: pd.DataFrame, test_size: int = 12) -> dict:
        # df should have columns 'ds' (date) and 'y' (value)
        # test_size is the number of month to use for testing
        if 'ds' not in df.columns or 'y' not in df.columns:
            raise ValueError("DataFrame must contain 'ds' and 'y' columns.")
        if self.forecast is None:
            raise ValueError("Model must be fitted and predictions made before evaluation.")
        
        train, test = train_test_split(df, test_size=test_size)
        self.train = train
        self.test = test

        test_true = test['y'].values
        self.train_pred = self.forecast['yhat'][:len(train)].values
        self.test_pred = self.forecast['yhat'][-len(test):].values

        mse = mean_squared_error(test_true, self.test_pred)
        mae = mean_absolute_error(test_true, self.test_pred)
        mape = mean_absolute_percentage_error(test_true, self.test_pred)

        return {
            'MSE': mse,
            'MAE': mae,
            'MAPE': mape
        }

    def plot(self):
        fig, ax = plt.subplots(figsize=(10, 6))

        # describe the training data
        if self.train is not None:
            ax.plot(self.train['ds'], self.train['y'], label='actural(Train)', color='blue')

        # describe the test data
        if self.test is not None:
            ax.plot(self.test['ds'], self.test['y'], label='actural(Test)', color='orange')

        # describe the training predictions
        if self.train_pred is not None and self.train is not None:
            ax.plot(self.train['ds'], self.train_pred, label='predicted(Train)', color='cyan', linestyle='--')

        # describe the test predictions
        if self.test_pred is not None and self.test is not None:
            ax.plot(self.test['ds'], self.test_pred, label='predicted(Test)', color='red', linestyle='--')

        ax.legend()
        ax.set_title('Prophet Forecast')

        # save the figure
        plt.tight_layout()
        plt.savefig('prophet_forecast.png')

        plt.show()