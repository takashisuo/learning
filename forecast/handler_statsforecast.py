from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
import pandas as pd
import sys
import numpy as np
import optuna
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
from pmdarima.model_selection import train_test_split

sys.path.append("../")
from forecast.handler_base import TimeSeriesAbstractHandler

# ---- StatsForecast Training Handler ----
class StatsForecastTrainingHandler(TimeSeriesAbstractHandler):
    def __init__(self, exog_cols=None, freq="M"):
        super().__init__(exog_cols)
        self.freq = freq
        self._train = None
        self._test = None
        self._train_pred = None
        self._test_pred = None
        self.best_params = None

    def fit(self, df: pd.DataFrame):
        # statsforecastは [unique_id, ds, y, (exog...)] の形式を要求
        if not set(["unique_id", "ds", "y"]).issubset(df.columns):
            raise ValueError("DataFrame must contain 'unique_id', 'ds', 'y' columns.")
        model = AutoARIMA(**(self.best_params if self.best_params else {}))
        self._model = StatsForecast(models=[model], freq=self.freq, n_jobs=-1)
        self._model.fit(df)

    def predict(self, periods: int, freq: str = None, future_exog: pd.DataFrame = None) -> pd.DataFrame:
        if self._model is None:
            raise ValueError("Model is not fitted. Call fit first.")
        fh = periods
        forecast = self._model.predict(h=fh, X_df=future_exog)
        self._forecast = forecast
        return forecast

    def tune_hyperparameters(
        self,
        df: pd.DataFrame,
        test_size: int = 12,
        val_size: int = 12,
        n_trials: int = 50,
    ) -> dict:
        train_df, test_df = train_test_split(df, test_size=test_size)
        self._train, self._test = train_df, test_df

        def objective(trial):
            params = {
                "season_length": trial.suggest_categorical("season_length", [1, 6, 12]),
                "d": trial.suggest_int("d", 0, 2),
                "D": trial.suggest_int("D", 0, 1),
                "max_p": trial.suggest_int("max_p", 2, 5),
                "max_q": trial.suggest_int("max_q", 2, 5),
            }
            tss = TimeSeriesSplit(test_size=val_size)
            cv_mse = []
            for train_idx, val_idx in tss.split(train_df):
                tr, val = train_df.iloc[train_idx], train_df.iloc[val_idx]
                model = StatsForecast(models=[AutoARIMA(**params)], freq=self.freq, n_jobs=-1)
                model.fit(tr)
                preds = model.predict(h=len(val))
                merged = val.merge(preds, on=["unique_id", "ds"], how="left")
                mse = mean_squared_error(merged["y"], merged["AutoARIMA"])
                cv_mse.append(mse)
            return np.mean(cv_mse)

        optuna.logging.disable_default_handler()
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, n_jobs=-1)
        self.best_params = study.best_params
        return self.best_params

    def evaluate(self, df: pd.DataFrame, test_size: int = 12) -> dict:
        if not set(["unique_id", "ds", "y"]).issubset(df.columns):
            raise ValueError("DataFrame must contain 'unique_id', 'ds', 'y' columns.")
        train, test = train_test_split(df, test_size=test_size)
        self._train, self._test = train, test
        self.fit(train)
        preds = self.predict(periods=len(test))
        merged = test.merge(preds, on=["unique_id", "ds"], how="left")
        mse = mean_squared_error(merged["y"], merged["AutoARIMA"])
        mae = mean_absolute_error(merged["y"], merged["AutoARIMA"])
        mape = mean_absolute_percentage_error(merged["y"], merged["AutoARIMA"])
        return {"MSE": mse, "MAE": mae, "MAPE": mape}

    def plot(self, save_path="statsforecast_train_test.png"):
        if self._forecast is None:
            raise ValueError("Run predict first.")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self._train["ds"], self._train["y"], label="Train", color="blue")
        ax.plot(self._test["ds"], self._test["y"], label="Test", color="orange")
        ax.plot(self._forecast["ds"], self._forecast["AutoARIMA"], label="Forecast", color="red")
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()


# ---- StatsForecast Product Handler ----
class StatsForecastProductHandler(TimeSeriesAbstractHandler):
    def __init__(self, exog_cols=None, model_params: dict = None, freq="M"):
        super().__init__(exog_cols)
        self.freq = freq
        self._model_params = model_params if model_params else {}
        self._train = None

    def fit(self, df: pd.DataFrame):
        if not set(["unique_id", "ds", "y"]).issubset(df.columns):
            raise ValueError("DataFrame must contain 'unique_id', 'ds', 'y' columns.")
        self._model = StatsForecast(models=[AutoARIMA(**self._model_params)], freq=self.freq, n_jobs=1)
        self._model.fit(df)
        self._train = df.copy()

    def predict(self, periods: int, freq: str = None, future_exog: pd.DataFrame = None) -> pd.DataFrame:
        if self._model is None:
            raise ValueError("Model is not fitted. Call fit first.")
        forecast = self._model.predict(h=periods, X_df=future_exog)
        self._forecast = forecast
        return forecast

    def evaluate(self, df: pd.DataFrame) -> dict:
        raise NotImplementedError("Evaluation is not implemented for production handler.")

    def plot(self, save_path="statsforecast_product.png"):
        if self._forecast is None:
            raise ValueError("No forecast to plot.")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self._train["ds"], self._train["y"], label="Train", color="blue")
        ax.plot(self._forecast["ds"], self._forecast["AutoARIMA"], label="Forecast", color="red")
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()