# %% [markdown]
# # SHAPによる時系列予測値の要因分解(Bike Sharing 日次 × LightGBM)
# 
# 処理は `src/pipeline.py` の各ステップに置き、このノートブックは呼び出しと解説だけを行う。SHAP の計算・分解・What-if は `xai_platform.ShapAnalyzer`(使い方は `src/xai_platform/README.md`)が担う。結論と注意事項は `outputs/report.md` を参照。
# 
# - 目的変数: `cnt`(1日のレンタル件数、単位: 件/日)。1日先(翌日)の予測。
# - 外生変数(気温・湿度・風速・天候)は **予測時点で既知(予報済み)と仮定**し、実績値を使っている。

# %%
import os, sys, warnings
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')
warnings.filterwarnings('ignore')
sys.path.insert(0, '..')
import pandas as pd
pd.set_option('display.max_columns', 30); pd.set_option('display.width', 200)
from src.pipeline import BikeAnalysis
from src.xai_platform import MANIFEST
a = BikeAnalysis()

# %% [markdown]
# ## 1. データ確認
# 訓練/テストは時間順に分ける(末尾20% = 2012-08-08〜12-31 がテスト期間)。ラグ28日分が必要なため、特徴量のある訓練期間は 2011-01-29 から始まる。`casual`・`registered` は cnt の内訳(リーク源)なので除外した。

# %%
a.step_data().show()

# %% [markdown]
# ## 2. 精度(Naive との比較)
# 目的変数の変換3種 × LightGBM / Ridge を TimeSeriesSplit(拡張窓)で調整し、テスト期間で評価する。説明には **LightGBM[level]**(cnt − 直近7日平均を学習)を使う。CV-RMSE が最良で、訓練期間を上回る水準も予測でき、SHAP が元の単位(件/日)のまま加法になるため。

# %%
a.step_baseline().show()

# %% [markdown]
# ## 3. SHAP の計算と検証
# 予測値 = ベースライン + Σ寄与 が両スケールで 1e-6 未満の誤差で成り立つこと(加法性)、Ridge で SHAP = 係数 × (x − 訓練平均) となることを確認する。

# %%
a.step_explain().show()

# %% [markdown]
# ## 4. 代表日の分解(ミクロ)
# 選定基準: テスト期間の中で、予測が最も高い日 / 最も低い日 / 前日比で予測が最も上昇した日 / 最も下落した日。参考として誤差が最大の日(ハリケーン・サンディ)も加えた。

# %%
a.step_micro().show()

# %% [markdown]
# ## 5. 時間方向の構成(マクロ)
# 訓練期間の寄与は学習データ上の説明(in-sample)である点に注意。

# %%
a.step_macro().show()

# %% [markdown]
# ## 6. 変化の要因分解(ΔSHAP)
# 予測値(t) − 予測値(s) = Σ_グループ[寄与(t) − 寄与(s)]。ベースラインは共通なので差で消える。

# %%
a.step_delta().show()

# %% [markdown]
# ## 7. What-if(外生変数のみを動かす)
# (A) Δ予測値はモデルの再予測、(B) 内訳は ΔSHAP。動かす範囲は訓練期間の 1〜99 パーセンタイル内に制限する(範囲外はクリップし、図表に警告を出す)。

# %%
a.step_whatif().show()

# %%
a.finalize()
MANIFEST.to_frame(a.out.root).head(60)


