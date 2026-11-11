"""学習と評価(回帰)。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .data import Dataset
from .provenance import produces_table


def fit(estimator, ds: Dataset):
    """scikit-learn 互換の推定器を訓練データで学習して返す。"""
    estimator.fit(ds.X_train, ds.y_train)
    return estimator


def evaluate(model, ds: Dataset) -> dict[str, float]:
    pred = model.predict(ds.X_test)
    return {
        "R2": r2_score(ds.y_test, pred),
        "RMSE": float(np.sqrt(mean_squared_error(ds.y_test, pred))),
        "MAE": mean_absolute_error(ds.y_test, pred),
    }


@produces_table
def evaluation_table(models: dict[str, object], ds: Dataset) -> pd.DataFrame:
    """複数モデルのテストデータでの R²・RMSE・MAE の表。"""
    rows = [{"モデル": name, **evaluate(m, ds)} for name, m in models.items()]
    return pd.DataFrame(rows).set_index("モデル")
