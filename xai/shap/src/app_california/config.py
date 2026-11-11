"""California Housing 分析固有の設定。"""
from __future__ import annotations

from pathlib import Path

from src.xai_platform import Unit
from src.xai_platform.io import OutputDirs

SEED = 42
TEST_SIZE = 0.2

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = OutputDirs(ROOT / "outputs")

TARGET_NAME = "MedHouseVal"
UNIT = Unit(label="10万ドル", alt_label="万ドル", alt_scale=10.0)  # 1.0 = 10万ドル
TARGET_CAP = 5.0  # 目的変数の上限打ち切り値

# 特徴量の日本語説明(初出時の説明表に使う)
FEATURE_DESCRIPTIONS = {
    "MedInc": "地区の世帯収入中央値(単位: 1万ドル)",
    "HouseAge": "地区の住宅築年数の中央値(年)",
    "AveRooms": "1世帯あたり平均部屋数",
    "AveBedrms": "1世帯あたり平均寝室数",
    "Population": "地区の人口",
    "AveOccup": "1世帯あたり平均居住人数",
    "Latitude": "緯度(大きいほど北)",
    "Longitude": "経度(大きいほど東=内陸)",
}

# ---------------------------------------------------------------- モデル
LGBM_PARAMS = dict(
    n_estimators=600,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=20,
    # 行サンプリングは使わない。使うと木の葉の件数が部分データ基準になり、
    # TreeExplainer の expected_value が「訓練データの平均予測」と一致しなくなるため。
    subsample=1.0,
    colsample_bytree=0.8,
    random_state=SEED,
    n_jobs=1,  # 再現性のため単スレッド
    verbose=-1,
)
RIDGE_ALPHA = 1.0

# ---------------------------------------------------------------- ミクロ
# 代表サンプル: {ファイル名タグ: (区分ラベル, 予測値の分位点)}
REPRESENTATIVES = {
    "high": ("予測が高い(95%点)", 0.95),
    "median": ("中央値付近(50%点)", 0.50),
    "low": ("予測が低い(5%点)", 0.05),
}
WHATIF_PCT_TARGET = "median"  # +10% の What-if を行う代表サンプル

# ---------------------------------------------------------------- What-if
BOUNDS_Q = (0.01, 0.99)  # 動かしてよい範囲 = 訓練データの 1〜99 パーセンタイル

# 複数変数シナリオ: {"ID 説明": {特徴量: (種類, 量)}}
SCENARIOS = {
    "S1 所得と広さが上がる": {"MedInc": ("sd", 1.0), "AveRooms": ("sd", 1.0)},
    "S2 築浅・世帯人数が少ない": {"HouseAge": ("abs", -10.0), "AveOccup": ("abs", -0.5)},
    "S3 海岸沿いに北西へ移動": {"Latitude": ("abs", 1.0), "Longitude": ("abs", -1.0)},
    "S4 3変数を同時に変更": {"MedInc": ("sd", 1.0), "HouseAge": ("abs", -10.0),
                         "AveOccup": ("abs", -0.5)},
}

# +10% を適用しない特徴量(緯度経度の10%は約3.5度で意味をなさない)
PCT_EXCLUDED = ("Latitude", "Longitude")

# 片方だけ動かすと非現実的になりうる相関ペア
CORRELATED_PAIRS = [("Latitude", "Longitude"), ("AveRooms", "AveBedrms")]

# ---------------------------------------------------------------- マクロ
TOP_DEPENDENCE = 6  # 依存プロットを描く上位特徴量数
# 外れ値が極端なため依存プロットの横軸を 1〜99% 点に絞る特徴量
DEPENDENCE_CLIP_X = ("AveOccup", "AveRooms", "AveBedrms", "Population")
INTERACTION_N = 1000  # 相互作用値のサブサンプル件数
TOP_INTERACTION_PAIRS = 3

# 全体構成の積み上げ図で使う特徴量グループ
FEATURE_GROUPS = {
    "所得": ["MedInc"],
    "立地": ["Latitude", "Longitude"],
    "住居": ["AveRooms", "AveBedrms", "AveOccup"],
    "築年": ["HouseAge"],
    "人口": ["Population"],
}
GROUP_COLORS = {"所得": "#2a78b5", "立地": "#e07b2a", "住居": "#3c9a5f",
                "築年": "#8b62b8", "人口": "#9a8a5a"}
N_PRED_BINS = 10
