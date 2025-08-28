#%%
import pandas as pd
import numpy as np
from statsforecast.core import StatsForecast
from statsforecast.models import AutoARIMA
from statsforecast.utils import AirPassengersDF

# サンプルデータとしてAirPassengersを使用
df = AirPassengersDF.rename(columns={'AirPassengers': 'y'})

# 外生変数 'x1', 'x2' を追加
# x1 は単純な増加トレンド
df['x1'] = np.arange(len(df)) + 100
# x2 はランダムなノイズを追加して、x1との相関を下げる
np.random.seed(42) # 再現性のためのシード値
df['x2'] = np.arange(len(df)) / 100 + np.random.randn(len(df)) * 0.5

# 予測に含める外生変数（未来のデータ）
future_df = pd.DataFrame({
    'unique_id': ['AirPassengers'] * 12,
    'ds': pd.date_range(start='1961-01-01', periods=12, freq='MS'),
    # 未来のx1データ
    'x1': np.arange(len(df), len(df) + 12) + 100,
    # 未来のx2データにもランダムなノイズを追加
    'x2': np.arange(len(df), len(df) + 12) / 100 + np.random.randn(12) * 0.5
})

# モデルの初期化
sf = StatsForecast(
    models=[AutoARIMA(season_length=12)],
    freq='MS'
)

# 予測を実行
Y_hat_df = sf.forecast(
    df=df,
    h=12,
    X_df=future_df
)

print("予測結果:")
print(Y_hat_df)
# %%
