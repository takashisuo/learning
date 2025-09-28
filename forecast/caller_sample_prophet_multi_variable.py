# time_series_handler.pyを参照する

#%%
import os, sys
import numpy as np
sys.path.append("../")
from forecast.helper import TimeSeriesHelper
from forecast.handler_prophet import ProphetTrainingHandler, ProphetProductHandler
import pandas as pd

#%%
# サンプルデータの読み込み(AirPassengersデータセット)
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
df = pd.read_csv(url)
df['Month'] = pd.to_datetime(df['Month'])
print(df.head())

# 1. フラグ変数 (0 or 1)
df['flag_var'] = np.random.choice([0, 1], size=len(df))

# 2. 数値データ (整数)
df['numeric_var'] = np.random.randint(0, 100, size=len(df))

# 3. カテゴリデータ (文字列)
categories = ['A', 'B', 'C']
df['category_var'] = np.random.choice(categories, size=len(df))
print(df["category_var"].value_counts())

# Prophet用にデータを変換
# なお、Prophetはカテゴリ変数を直接扱えないため、One-Hot Encodingなどで数値化が必要
# 今回はOne-Hot Encodingを行い、'category_var_A', 'category_var_B', 'category_var_C'の3列を追加
# exog_colsにこれらを指定するが、category_var_Aは存在しないため、category_var_B, category_var_Cのみ指定
df = pd.get_dummies(df, columns=['category_var'], drop_first=False)
exog_cols = ['flag_var', 'numeric_var', "category_var_B", "category_var_C"]
print(df.head())

#%%
# Prophet用にデータを変換
df = TimeSeriesHelper.to_prophet_format(df,
                                        date_col='Month',
                                        value_col='Passengers',
                                        exog_cols=exog_cols)
future_periods = 12
ph = ProphetTrainingHandler(exog_cols=exog_cols)

# ベストパラメータ探索
best_params = ph.tune_hyperparameters(df, test_size=future_periods, n_trials=20)

# ベストモデルで学習
ph.build_best_model(best_params)

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
ph_prod = ProphetProductHandler(model_params=best_params, exog_cols=exog_cols)
ph_prod.fit(df)
forecast_df_prod = ph_prod.predict(periods=future_periods,
                                   freq='M',
                                   future_exog=df[exog_cols].tail(future_periods))
print(forecast_df_prod[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
ph_prod.plot()