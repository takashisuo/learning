"""SQL Serverアップロード検証用のダミーデータ（5カラム）を生成する。"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl

_BASE_DATE = datetime(2020, 1, 1)
_SECONDS_PER_YEAR = 86_400 * 365


def generate_dataframe(n_rows: int, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    df = pl.DataFrame(
        {
            "id": np.arange(n_rows, dtype=np.int64),
            "category_code": rng.integers(0, 1_000, n_rows, dtype=np.int32),
            "amount": rng.uniform(0, 100_000, n_rows).round(2),
            "status": rng.integers(0, 3, n_rows, dtype=np.int8),
            "seconds_offset": rng.integers(0, _SECONDS_PER_YEAR, n_rows, dtype=np.int64),
        }
    )
    return df.with_columns(
        (pl.lit(_BASE_DATE) + pl.duration(seconds=pl.col("seconds_offset"))).alias("updated_at")
    ).drop("seconds_offset")
