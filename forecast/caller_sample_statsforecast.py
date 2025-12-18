# time_series_handler.pyを参照する

#%%
import os, sys
sys.path.append("../")
from forecast.helper import TimeSeriesHelper
from forecast.handler_statsforecast import StatsForecastTrainingHandler, StatsForecastProductHandler
import pandas as pd

# サンプルデータの読み込み(AirPassengersデータセット)
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
df = pd.read_csv(url)
df['Month'] = pd.to_datetime(df['Month'])
print(f"---- head ----")
print(df.head())
print(f"---- tail ----")
print(df.tail())

#%%
# statsforecast用にデータを変換
df_stats = TimeSeriesHelper.to_statsforecast_format(df, date_col='Month', value_col='Passengers')
future_periods = 12
ph = StatsForecastTrainingHandler()

ph.fit(df_stats)

result = ph.predict(periods=12)

print(f"result: {result}")

#%%
# ベストパラメータ探索

df_stats = TimeSeriesHelper.to_statsforecast_format(df, date_col='Month', value_col='Passengers')
future_periods = 12
ph = StatsForecastTrainingHandler()

best_params = ph.tune_hyperparameters(df_stats, test_size=future_periods, n_trials=20)

# ベストモデルで学習
ph.build_best_model(best_params)

# 評価
eval_results = ph.evaluate(df_stats, test_size=future_periods)
print(eval_results)

# 可視化
ph.plot()

# モデル保存
ph.save_model("prophet_training_model.pkl")
train_df = ph.get_train_data()

#%%
# 本番用ハンドラの使用例
ph_prod = StatsForecastProductHandler(model_params=best_params)
ph_prod.fit(df_stats)
forecast_df_prod = ph_prod.predict(periods=future_periods, freq='M')
print(forecast_df_prod[['ds', 'y']].tail())
ph_prod.plot() 
#"""