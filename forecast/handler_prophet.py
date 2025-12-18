from prophet import Prophet
import numpy as np
import os, sys
import joblib
import pandas as pd
import optuna
import logging
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from pmdarima.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
sys.path.append("../")
from forecast.helper import TimeSeriesHelper
from forecast.handler_base import TimeSeriesAbstractHandler

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
            print(f"added regressor: {col}")
        self._forecast = None
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

        if (self.exog_cols is not None and len(self.exog_cols) > 0) and future_exog is None:
            raise ValueError("Exogenous columns are specified but future_exog is None.")
        elif (self.exog_cols is None and len(self.exog_cols) == 0) and future_exog is not None:
            raise ValueError("Exogenous columns are not specified but future_exog is provided.")

        if future_exog is not None:
            # future_exogが提供されている場合
            future_exog = future_exog.copy()
            
            # dsカラムがない場合は追加
            if 'ds' not in future_exog.columns:
                # future_exogの行数がperiodsと一致する場合、未来のdateを設定
                if len(future_exog) == periods:
                    future_exog['ds'] = future['ds'].tail(periods).values
                else:
                    # 全期間のdateを設定
                    future_exog['ds'] = future['ds'].values[:len(future_exog)]
            
            # 外生変数の値を設定
            for col in self.exog_cols:
                if col in future_exog.columns:
                    # NaN値は訓練データの最後の値で補完
                    if future_exog[col].isna().any():
                        future_exog[col].fillna(self._train[col].iloc[-1], inplace=True)
                    future[col] = None  # 初期化
                else:
                    # カラムが存在しない場合は訓練データの最後の値で埋める
                    future_exog[col] = self._train[col].iloc[-1]
            
            # futureデータフレームに外生変数をマージ
            # 最初に既存の外生変数カラムを削除
            for col in self.exog_cols:
                if col in future.columns:
                    future = future.drop(columns=[col])
            
            future = future.merge(future_exog[['ds'] + self.exog_cols], on='ds', how='left')
            
            # マージ後にNaN値が残っている場合は補完
            for col in self.exog_cols:
                if future[col].isna().any():
                    future[col].fillna(self._train[col].iloc[-1], inplace=True)

        self._forecast = self._model.predict(future)
        return self._forecast
    
    def tune_hyperparameters(
            self,
            df: pd.DataFrame,
            test_size: int = 12,
            val_size: int = 12,
            n_trials: int = 100,
            n_jobs: int = -1,
            freq: str = "M",
            params: dict = None
        ) -> dict:
        """
        Optunaを使ってProphetのハイパーパラメータ探索を行い、最適パラメータを返す。

        Returns:
            dict: 最適パラメータ {'changepoint_prior_scale': ..., 'seasonality_prior_scale': ...}
        """
        train_df, test_df = train_test_split(df, test_size=test_size)
        self._train = train_df
        self._test = test_df

        def objective(trial):

            if params is None:
                optuna_params = {
                    'changepoint_prior_scale': trial.suggest_float('changepoint_prior_scale', 0.001, 0.5),
                    'seasonality_prior_scale': trial.suggest_float('seasonality_prior_scale', 0.01, 10.0),
                    'seasonality_mode': trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative']),
                    'changepoint_range': trial.suggest_float('changepoint_range', 0.8, 0.95, step=0.001),
                    'n_changepoints': trial.suggest_int('n_changepoints', 20, 35)
                }
            else:
                optuna_params = params

            tss = TimeSeriesSplit(test_size=val_size)
            cv_mse = []

            for fold, (train_index, val_index) in enumerate(tss.split(train_df)):

                train_data = train_df.iloc[train_index]
                val_data = train_df.iloc[val_index]
            
                model = Prophet(
                    **optuna_params
                )

                for col in self.exog_cols:
                    model.add_regressor(col)

                model.fit(train_data)

                future = model.make_future_dataframe(periods=len(val_data), freq=freq)
                if self.exog_cols:
                    # 外生変数の予測時の値を最後の値で仮固定
                    for col in self.exog_cols:
                        last_val = train_data[col].iloc[-1]  # 最後の値を取得
                        future[col] = last_val
                
                forecast = model.predict(future)
                preds = forecast.tail(len(val_data))['yhat'].values
                val_mse = mean_squared_error(val_data['y'].values, preds)
                cv_mse.append(val_mse)

            return np.mean(cv_mse)

        optuna.logging.disable_default_handler()
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

        self.best_params = study.best_params
        print(f"Best params: {self.best_params}")
        return self.best_params
    
    def build_best_model(self, params: dict):
        """
        tune_hyperparametersで探索済みの最適パラメータを使って学習済みモデルを作成。
        """
        if not hasattr(self, 'best_params'):
            raise ValueError("Best params not found. Run tune_hyperparameters first.")

        self._model = Prophet(
            **params
        )
        if self.exog_cols:
            for col in self.exog_cols:
                self._model.add_regressor(col)

        self.fit(self._train)
        return self._model

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
        
        if self._train is None or self._test is None:
            train, test = train_test_split(df, test_size=test_size)
            self._train = train
            self._test = test
        else:
            train = self._train
            test = self._test

        # 外生変数の処理を改善
        future_exog = None
        print(f"self.exog_cols: {self.exog_cols}")
        if self.exog_cols:
            # テストデータの外生変数を準備
            future_exog = test[self.exog_cols].copy()
            
            # 訓練データの最後の値を使ってNaN値を補完
            for col in self.exog_cols:
                if future_exog[col].isna().any():
                    last_val = train[col].iloc[-1]
                    future_exog[col].fillna(last_val, inplace=True)

        test_forecast: pd.DataFrame = self.predict(
            periods=test_size,
            freq=freq,
            future_exog=future_exog
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
    def __init__(self, exog_cols=None, model_params: dict = None):
        """ Initializes the Prophet model and necessary attributes.
        Prophet requires a DataFrame with two columns: 'ds' for the date and 'y' for the value to be forecasted.
        """
        super().__init__(exog_cols)
        self._model_params = model_params if model_params is not None else {}
        self._model = Prophet(**self._model_params)
        print(f"model params: {self._model_params}")
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

        if (self.exog_cols is not None and len(self.exog_cols) > 0) and future_exog is None:
            raise ValueError("Exogenous columns are specified but future_exog is None.")
        elif (self.exog_cols is None and len(self.exog_cols) == 0) and future_exog is not None:
            raise ValueError("Exogenous columns are not specified but future_exog is provided.")

        if future_exog is not None:
            # future_exogが提供されている場合
            future_exog = future_exog.copy()
            
            # dsカラムがない場合は追加
            if 'ds' not in future_exog.columns:
                # future_exogの行数がperiodsと一致する場合、未来のdateを設定
                if len(future_exog) == periods:
                    future_exog['ds'] = future['ds'].tail(periods).values
                else:
                    # 全期間のdateを設定
                    future_exog['ds'] = future['ds'].values[:len(future_exog)]
            
            # 外生変数の値を設定
            for col in self.exog_cols:
                if col in future_exog.columns:
                    # NaN値は訓練データの最後の値で補完
                    if future_exog[col].isna().any():
                        future_exog[col].fillna(self._train[col].iloc[-1], inplace=True)
                    future[col] = None  # 初期化
                else:
                    # カラムが存在しない場合は訓練データの最後の値で埋める
                    future_exog[col] = self._train[col].iloc[-1]
            
            # futureデータフレームに外生変数をマージ
            # 最初に既存の外生変数カラムを削除
            for col in self.exog_cols:
                if col in future.columns:
                    future = future.drop(columns=[col])
            
            future = future.merge(future_exog[['ds'] + self.exog_cols], on='ds', how='left')
            
            # マージ後にNaN値が残っている場合は補完
            for col in self.exog_cols:
                if future[col].isna().any():
                    future[col].fillna(self._train[col].iloc[-1], inplace=True)

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