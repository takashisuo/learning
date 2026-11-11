"""このアプリで使うモデルの定義。"""
from __future__ import annotations

from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

from src.xai_platform.data import Dataset
from src.xai_platform.model import fit

from .config import LGBM_PARAMS, RIDGE_ALPHA, SEED


def train_lgbm(ds: Dataset, **overrides) -> LGBMRegressor:
    return fit(LGBMRegressor(**{**LGBM_PARAMS, **overrides}), ds)


def train_ridge(ds: Dataset, alpha: float = RIDGE_ALPHA) -> Ridge:
    """SHAP 検証用の線形モデル(標準化はしない: 係数×(x−平均) の比較を単純にするため)。"""
    return fit(Ridge(alpha=alpha, random_state=SEED), ds)
