"""学習・評価。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .data import SEED, Dataset

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


def train_lgbm(ds: Dataset, **overrides) -> LGBMRegressor:
    params = {**LGBM_PARAMS, **overrides}
    model = LGBMRegressor(**params)
    model.fit(ds.X_train, ds.y_train)
    return model


def train_ridge(ds: Dataset, alpha: float = 1.0) -> Ridge:
    """SHAP 検証用の線形モデル(標準化はしない: 係数×(x−平均) の比較を単純にするため)。"""
    model = Ridge(alpha=alpha, random_state=SEED)
    model.fit(ds.X_train, ds.y_train)
    return model


def evaluate(model, ds: Dataset) -> dict[str, float]:
    pred = model.predict(ds.X_test)
    return {
        "R2": r2_score(ds.y_test, pred),
        "RMSE": float(np.sqrt(mean_squared_error(ds.y_test, pred))),
        "MAE": mean_absolute_error(ds.y_test, pred),
    }


def evaluation_table(models: dict[str, object], ds: Dataset) -> pd.DataFrame:
    rows = [{"モデル": name, **evaluate(m, ds)} for name, m in models.items()]
    return pd.DataFrame(rows).set_index("モデル")
