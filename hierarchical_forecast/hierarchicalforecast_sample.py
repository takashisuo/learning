#%%
# !pip install -U numba statsforecast datasetsforecast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# compute base forecast no coherent
from statsforecast.core import StatsForecast
from statsforecast.models import AutoARIMA, Naive

# obtain hierarchical reconciliation methods and evaluation
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.evaluation import evaluate
from hierarchicalforecast.methods import BottomUp, TopDown, MiddleOut
from utilsforecast.losses import mse

def create_hierarchical_forecast_sample_data():
    """階層時系列予測のオリジナルサンプルデータを生成"""
    print("=== 1. オリジナル階層データの生成 ===")
    
    # 時系列の期間設定（5年間の四半期データ）
    start_date = '2019-03-31'
    periods = 20  # 5年 × 4四半期
    dates = pd.date_range(start=start_date, periods=periods, freq='QE')
    
    # 階層構造の定義
    # Level 1: 全体（合計）
    # Level 2: 地域（東日本マケ, 西日本マケ）
    # Level 3: 地域×製品（東日本マケ_支社A, 東日本マケ_支社B, 西日本マケ_支社C, 西日本マケ_支社D）
    
    # ベースとなる時系列パターンを生成
    np.random.seed(42)
    
    # 季節性のあるトレンドを生成する関数
    def generate_base_series(base_level=100, trend=0.02, seasonal_amplitude=20, noise_level=5):
        seasonal_pattern = seasonal_amplitude * np.sin(2 * np.pi * np.arange(periods) / 4)
        trend_component = base_level * (1 + trend) ** np.arange(periods)
        noise = np.random.normal(0, noise_level, periods)
        return trend_component + seasonal_pattern + noise
    
    # 最下位レベル（個別系列）のデータを生成
    series_data = {}
    
    # 東日本マケ_支社A: 高成長、強い季節性
    series_data['東日本マケ_支社A'] = generate_base_series(
        base_level=80, trend=0.03, seasonal_amplitude=15, noise_level=4
    )
    
    # 東日本マケ_支社B: 中成長、中程度の季節性
    series_data['東日本マケ_支社B'] = generate_base_series(
        base_level=120, trend=0.015, seasonal_amplitude=25, noise_level=6
    )
    
    # 西日本マケ_支社C: 低成長、弱い季節性
    series_data['西日本マケ_支社C'] = generate_base_series(
        base_level=60, trend=0.01, seasonal_amplitude=10, noise_level=3
    )
    
    # 西日本マケ_支社D: 高成長、強い季節性
    series_data['西日本マケ_支社D'] = generate_base_series(
        base_level=90, trend=0.025, seasonal_amplitude=20, noise_level=5
    )
    
    # 階層の上位レベルを計算
    # 地域レベル
    series_data['東日本マケ'] = series_data['東日本マケ_支社A'] + series_data['東日本マケ_支社B']
    series_data['西日本マケ'] = series_data['西日本マケ_支社C'] + series_data['西日本マケ_支社D']
    
    # 全体レベル
    series_data['合計'] = series_data['東日本マケ'] + series_data['西日本マケ']
    
    # Y_df（時系列データフレーム）の作成
    y_data_list = []
    for unique_id, values in series_data.items():
        for i, value in enumerate(values):
            y_data_list.append({
                'unique_id': unique_id,
                'ds': dates[i],
                'y': max(0, value)  # 負の値を避ける
            })
    
    Y_df = pd.DataFrame(y_data_list)
    Y_df['ds'] = pd.to_datetime(Y_df['ds'])
    
    # S_df（階層構造行列）の作成
    # 階層予測では、最下位レベル（bottom level）の系列のみを基準とする
    # 最下位レベル系列
    bottom_level_series = ['東日本マケ_支社A', '東日本マケ_支社B', '西日本マケ_支社C', '西日本マケ_支社D']
    # 全ての系列（階層順: 上位レベルから下位レベル）
    all_series = ['合計', '東日本マケ', '西日本マケ', '東日本マケ_支社A', '東日本マケ_支社B', '西日本マケ_支社C', '西日本マケ_支社D']
    
    # S行列の作成：各行は上位レベル系列が最下位レベル系列の線形結合であることを表現
    s_matrix = []
    
    # 合計行: 全ての最下位レベル系列の合計
    s_matrix.append([1, 1, 1, 1])  # 東日本マケ_支社A + 東日本マケ_支社B + 西日本マケ_支社C + 西日本マケ_支社D
    
    # 東日本マケ行: 東日本マケ地域の製品の合計
    s_matrix.append([1, 1, 0, 0])  # 東日本マケ_支社A + 東日本マケ_支社B
    
    # 西日本マケ行: 西日本マケ地域の製品の合計
    s_matrix.append([0, 0, 1, 1])  # 西日本マケ_支社C + 西日本マケ_支社D
    
    # 最下位レベル行: 単位行列（各系列は自分自身のみ）
    s_matrix.append([1, 0, 0, 0])  # 東日本マケ_支社A
    s_matrix.append([0, 1, 0, 0])  # 東日本マケ_支社B
    s_matrix.append([0, 0, 1, 0])  # 西日本マケ_支社C
    s_matrix.append([0, 0, 0, 1])  # 西日本マケ_支社D
    
    S_df = pd.DataFrame(s_matrix, 
                        index=all_series, 
                        columns=bottom_level_series)
    S_df = S_df.reset_index(names="unique_id")
    
    print("\n階層構造行列（S行列）の説明:")
    print("  - 行: 全ての系列（上位→下位の順）")
    print("  - 列: 最下位レベル系列のみ")
    print("  - 値: 最下位レベル系列の線形結合係数")
    print("  - 最下部4x4は単位行列になっている")
    
    # tagsの作成（階層レベルの定義）
    tags = {
        '合計': ['合計'],
        'Region': ['東日本マケ', '西日本マケ'],
        'Region/Product': ['東日本マケ_支社A', '東日本マケ_支社B', '西日本マケ_支社C', '西日本マケ_支社D']
    }
    
    print(f"時系列データの形状: {Y_df.shape}")
    print(f"階層構造行列の形状: {S_df.shape}")
    print(f"階層レベル: {list(tags.keys())}")
    print("\n時系列データの最初の5行:")
    print(Y_df.head())

    print("\n階層構造行列の最初の5行:")
    print(S_df.head())

    print("\n各階層レベルの系列数:")
    for level, series_list in tags.items():
        print(f"  {level}: {len(series_list)}系列")
    
    # データの統計サマリー
    print("\n=== データの統計サマリー ===")
    summary_stats = Y_df.groupby('unique_id')['y'].agg(['mean', 'std', 'min', 'max']).round(2)
    print(summary_stats)
    
    return Y_df, S_df, tags

def create_hierarchical_forecast_sample_data_with_exog():
    # 元の関数をコピー
    Y_df, S_df, tags = create_hierarchical_forecast_sample_data()

    print("\n\n=== 2. 外生変数の追加 ===")

    # ---------------------------------------------
    # 既存のY_dfに外生変数を追加する
    # ---------------------------------------------

    # 各unique_idに対応する広告費と祝日データを生成
    exog_data = []

    # 時系列の期間設定（過去データ + 予測期間）
    # 元のデータは2019-03-31から2023-12-31までの20四半期
    # 予測期間を4四半期（1年間）追加する
    future_periods = 4
    all_dates = pd.date_range(start='2019-03-31', periods=20 + future_periods, freq='QE')

    # 各unique_idに対して外生変数を生成
    np.random.seed(43)
    for uid in Y_df['unique_id'].unique():
        uid_df = Y_df[Y_df['unique_id'] == uid].copy()
        
        # 広告費 (ランダムな連続値)
        # 将来の期間も予測して追加
        adv_spend_past = np.random.uniform(10, 50, len(uid_df))
        adv_spend_future = np.random.uniform(15, 60, future_periods)
        adv_spend = np.concatenate([adv_spend_past, adv_spend_future])
        
        # 祝日 (ランダムなバイナリ値)
        # 将来の期間も予測して追加
        is_holiday_past = np.random.choice([0, 1], len(uid_df), p=[0.9, 0.1])
        is_holiday_future = np.random.choice([0, 1], future_periods, p=[0.8, 0.2])
        is_holiday = np.concatenate([is_holiday_past, is_holiday_future])

        # データフレームに整形
        exog_df = pd.DataFrame({
            'unique_id': uid,
            'ds': all_dates,
            'is_holiday': is_holiday,
            'advertisement_spend': adv_spend
        })
        exog_data.append(exog_df)

    # 全ての外生変数を一つのデータフレームに結合
    Exog_df = pd.concat(exog_data, ignore_index=True)

    # 元のY_dfと外生変数データフレームを結合
    Y_df_with_exog = Y_df.merge(Exog_df, on=['ds', 'unique_id'], how='left')

    print("\n新しい時系列データセットの最初の5行 (外生変数を含む):")
    print(Y_df_with_exog.head())

    print("\n外生変数データセットの最初の5行 (未来のデータを含む):")
    print(Exog_df.head())

    # 将来の予測期間のためのデータフレームも準備
    # Prophetに渡すためのfuture_df
    future_df = Exog_df[Exog_df['ds'].isin(all_dates[20:])].copy()

    print("\n将来予測用データセットの最初の5行:")
    print(future_df.head())

    return Y_df_with_exog, S_df, tags, Exog_df, future_df

# 関数を実行して結果を確認
#Y_df, S_df, tags, Exog_df, future_df = create_hierarchical_forecast_sample_data_with_exog()

Y_df, S_df, tags = create_hierarchical_forecast_sample_data()
print(f"Y_dfの形状: {Y_df.shape}")
try:
    display(Y_df.head())
except:
    print(Y_df.head())
print(f"S_dfの形状: {S_df.shape}")
try:
    display(S_df.head())
except:
    print(S_df.head())
print(f"tagsの内容: {tags}")

#%%

print("\n=== 2. 訓練・テストデータの分割 ===")
# split train/test sets (最後の4期間をテスト用)
Y_test_df = Y_df.groupby('unique_id').tail(4)
Y_train_df = Y_df.drop(Y_test_df.index)

print(f"訓練データ: {Y_train_df.shape[0]}観測値")
print(f"テストデータ: {Y_test_df.shape[0]}観測値")
print(f"予測期間: 4四半期")

#%%
print("\n=== 3. ベース予測の計算 ===")
# Compute base auto-ARIMA predictions
fcst = StatsForecast(models=[AutoARIMA(season_length=4), Naive()],
                     freq='QE', n_jobs=1)

print("使用モデル:")
print("  - AutoARIMA: 自動でARIMAパラメータを選択")
print("  - Naive: 前期の値をそのまま予測値とする")

Y_hat_df = fcst.forecast(df=Y_train_df, h=4)

print(f"\nベース予測結果の形状: {Y_hat_df.shape}")
print("ベース予測の最初の5行:")
print(Y_hat_df.head())

print("\n=== 4. 階層調整（Reconciliation）の実行 ===")
# Reconcile the base predictions
reconcilers = [
    BottomUp(),  # ボトムアップ: 最下位レベルの予測を集約
    TopDown(method='forecast_proportions'),  # トップダウン: 最上位レベルから比例配分
    # MiddleOut(middle_level='Region',  # ミドルアウト: 中間レベル（Region）から上下に展開
    #           top_down_method='forecast_proportions')
]

print("使用する調整手法:")
print("  1. BottomUp: 最下位レベル(個別系列)の予測値を上位レベルに集約")
print("  2. TopDown: 最上位レベルの予測を下位レベルに比例配分")
# print("  3. MiddleOut: 中間レベル(Region)から上下に展開")
print("  注意: MiddleOutは一時的に無効化（デバッグのため）")

print(f"\nデバッグ情報:")
print(f"tags辞書の内容: {tags}")
print(f"tags辞書のキー: {list(tags.keys())}")

hrec = HierarchicalReconciliation(reconcilers=reconcilers)
Y_rec_df = hrec.reconcile(Y_hat_df=Y_hat_df, Y_df=Y_train_df,
                          S=S_df, tags=tags)

print(f"\n調整後予測結果の形状: {Y_rec_df.shape}")
print("調整後予測の列名:", Y_rec_df.columns.tolist())

print("\n=== 5. 予測結果の詳細表示 ===")
# まず実際の列名を確認
print("調整後予測データの実際の列名:")
print(Y_rec_df.columns.tolist())

# 特定の系列の予測結果を比較
sample_series = Y_rec_df['unique_id'].iloc[0]
sample_forecast = Y_rec_df[Y_rec_df['unique_id'] == sample_series]

print(f"\n系列 '{sample_series}' の予測結果:")
# 存在する列名のみを使用
available_cols = ['ds', 'unique_id']
for col in Y_rec_df.columns:
    if 'AutoARIMA' in col and col not in available_cols:
        available_cols.append(col)

print("表示する列:", available_cols)
display_cols = available_cols[:6]  # 最初の6列まで表示
print(sample_forecast[display_cols].round(2))

print("\n=== 6. 予測精度の評価 ===")
# Merge with actual test data
df = Y_rec_df.merge(Y_test_df, on=['unique_id', 'ds'])

print("実績値との比較データの形状:", df.shape)
print("\n実績値を含む結果の最初の5行:")
# 存在する列のみを表示
base_cols = ['unique_id', 'ds', 'y']
arima_cols = [col for col in df.columns if 'AutoARIMA' in col][:2]  # 最初の2つのAutoARIMA関連列
display_cols = base_cols + arima_cols
print(df[display_cols].head())

# Evaluate forecasts
evaluation = evaluate(df=df,
                      tags=tags,
                      train_df=Y_train_df,
                      metrics=[mse],
                      benchmark="Naive")

print("\n=== 7. 評価結果 ===")
arima_results = evaluation.set_index(["level", "metric"]).filter(like="ARIMA", axis=1)
print("AutoARIMAモデルの平均二乗誤差(MSE):")
print(arima_results.round(4))

print("\n各手法の説明:")
print("実際の調整手法（列名から判断）:")
arima_columns = [col for col in Y_rec_df.columns if 'AutoARIMA' in col]
for i, col in enumerate(arima_columns):
    print(f"  {i+1}. {col}")

print("\n調整手法の理論:")
print("  - BottomUp: 最下位レベル(個別系列)の予測値を上位レベルに集約")
print("  - TopDown: 最上位レベルの予測を下位レベルに比例配分")
print("  - MiddleOut: 中間レベルから上下に展開")

print("\n=== 8. 階層レベル別の性能比較 ===")
print("\n評価結果の構造を確認:")
print("evaluation データの形状:", evaluation.shape)
print("evaluation のindex:", evaluation.index.names)
print("evaluation の列名:", evaluation.columns.tolist())
print("\nevaluationの最初の5行:")
print(evaluation.head())

print("\narima_resultsの構造:")
print("arima_results の形状:", arima_results.shape)
print("arima_results のindex:", arima_results.index.names)
if len(arima_results.index.names) >= 2:
    print("利用可能なlevel値:", arima_results.index.get_level_values('level').unique())
    if len(arima_results.index.names) >= 2:
        print("利用可能なmetric値:", arima_results.index.get_level_values(arima_results.index.names[1]).unique())

print("\n階層レベル別MSE比較:")
try:
    if 'level' in arima_results.index.names:
        for level in arima_results.index.get_level_values('level').unique():
            level_results = arima_results.loc[level] if len(arima_results.index.names) > 1 else arima_results
            print(f"\n{level}レベル:")
            print(level_results.round(4))
            
            # 最良の手法を特定（メトリック次元がある場合のみ）
            if len(level_results.shape) > 1 and level_results.shape[0] > 0:
                # メトリック行が存在する場合
                metric_index = level_results.index[0] if len(level_results.index) > 0 else None
                if metric_index is not None:
                    metric_row = level_results.iloc[0]  # 最初の行（通常MSE）
                    best_method = metric_row.idxmin()
                    best_score = metric_row.min()
                    print(f"  最良手法: {best_method} (スコア: {best_score:.4f})")
            elif len(level_results.shape) == 1:
                # 1次元の場合
                best_method = level_results.idxmin()
                best_score = level_results.min()
                print(f"  最良手法: {best_method} (スコア: {best_score:.4f})")
    else:
        print("レベル情報が見つかりません。全体結果:")
        print(arima_results.round(4))
        
except Exception as e:
    print(f"階層別比較でエラー: {e}")
    print("代替表示:")
    print(arima_results.round(4))

print("\n=== 9. 系列別予測結果サンプル ===")
# Show detailed forecast for a few series
sample_series_list = Y_rec_df['unique_id'].unique()[:3]
for series in sample_series_list:
    print(f"\n--- {series} の予測結果 ---")
    # 存在する列のみを使用
    base_cols = ['ds', 'y']
    arima_cols = [col for col in df.columns if 'AutoARIMA' in col][:2]
    available_cols = base_cols + arima_cols
    
    series_data = df[df['unique_id'] == series][available_cols].round(2)
    print(series_data.to_string(index=False))

print("\n=== 10. オリジナルデータの特徴説明 ===")
print("生成されたデータの特徴:")
print("  - 期間: 2019年Q1〜2023年Q4（5年間、20四半期）")
print("  - 階層構造:")
print("    * Level 1 (合計): 全体")
print("    * Level 2 (Region): 東日本マケ, 西日本マケ")
print("    * Level 3 (Region/Product): 東日本マケ_支社A, 東日本マケ_支社B, 西日本マケ_支社C, 西日本マケ_支社D")
print("  - データ特性:")
print("    * 季節性: 四半期パターン")
print("    * トレンド: 各系列で異なる成長率")
print("    * ノイズ: 適度なランダムな変動")
print("    * 階層制約: 下位レベルの合計が上位レベルと一致")