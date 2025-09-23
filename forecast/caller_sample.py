# time_series_handler.pyを参照する

#%%
import os, sys
sys.path.append("../")
from forecast.time_series_handler import TimeSeriesAbstractHandler, ProphetTrainingHandler, ProphetProductHandler
import pandas as pd

#%%
# サンプルデータの読み込み(AirPassengersデータセット)
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
df = pd.read_csv(url)
df['Month'] = pd.to_datetime(df['Month'])
print(df.head())

# Prophet用にデータを変換
ph = ProphetTrainingHandler()
df = ph.transform_to_prophet_format(df, date_col='Month', value_col='Passengers')
future_periods = 12

# 評価
eval_results = ph.evaluate(df, test_size=future_periods)
print(eval_results)

# 可視化
ph.plot()

# モデル保存
ph.save_model("prophet_training_model.pkl")
train_df = ph.get_train_data()

#%%
# 本番用ハンドラの使用例
ph_prod = ProphetProductHandler()
ph_prod.fit(df)
forecast_df_prod = ph_prod.predict(periods=future_periods, freq='M')
print(forecast_df_prod[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
ph_prod.plot() 