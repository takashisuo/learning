# time_series_handler.pyを参照する

#%%
import os, sys
sys.path.append("../")
from forecast.time_series_handler import TimeSeriesAbstractHandler, ProphetHandler
import pandas as pd

#%%
# サンプルデータの読み込み(AirPassengersデータセット)
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
df = pd.read_csv(url)
df['Month'] = pd.to_datetime(df['Month'])
print(df.head())

# Prophet用にデータを変換
ph = ProphetHandler()
df = ph.transform_to_prophet_format(df, date_col='Month', value_col='Passengers')

# モデルの学習
ph.fit(df)

# 予測
future_periods = 12
forecast_df = ph.predict(periods=future_periods, freq='M')
print(forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# 評価
eval_results = ph.evaluate(df, test_size=future_periods)
print(eval_results)

# 可視化
ph.plot()