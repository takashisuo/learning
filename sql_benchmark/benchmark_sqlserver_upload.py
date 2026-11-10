"""pandas/polarsからSQL Serverへの大量データアップロード方式を比較するベンチマーク。

検証対象:
  - pandas.to_sql        : デフォルト executemany / method="multi" / fast_executemany
  - polars.write_database : デフォルト(sqlalchemy engine) / fast_executemany
  - pyodbc 生 executemany : fast_executemany 有無

ADBC engine は SQL Server 用ドライバが存在しないため対象外。
大量行のバルクロードは bcp_load.py（BULK INSERT系）を参照。

実行例:
  python benchmark_sqlserver_upload.py --rows 200000 --methods pandas_default,pandas_fast_executemany,pyodbc_fast_executemany
  python benchmark_sqlserver_upload.py --list-drivers
"""
from __future__ import annotations

import argparse
import os
import time
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
import polars as pl
import pyodbc
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from db_config import SqlServerConfig
from sample_data import generate_dataframe

CREATE_TABLE_SQL = """
CREATE TABLE {table} (
    id BIGINT,
    category_code INT,
    amount FLOAT,
    status SMALLINT,
    updated_at DATETIME2
)
"""
INSERT_SQL_TEMPLATE = (
    "INSERT INTO {table} (id, category_code, amount, status, updated_at) VALUES (?, ?, ?, ?, ?)"
)


@dataclass
class BenchmarkResult:
    method: str
    rows: int
    seconds: float | None
    error: str | None = None

    @property
    def rows_per_sec(self) -> float | None:
        if self.seconds is None or self.seconds <= 0:
            return None
        return self.rows / self.seconds


def recreate_table(base_engine: Engine, table: str) -> None:
    with base_engine.begin() as conn:
        conn.execute(text(f"IF OBJECT_ID('{table}', 'U') IS NOT NULL DROP TABLE {table}"))
        conn.execute(text(CREATE_TABLE_SQL.format(table=table)))


def method_pandas_default(*, config: SqlServerConfig, table: str, chunksize: int, pdf: pd.DataFrame, **_) -> None:
    engine = create_engine(config.sqlalchemy_url())
    try:
        pdf.to_sql(table, engine, if_exists="append", index=False, chunksize=chunksize)
    finally:
        engine.dispose()


def method_pandas_multi(*, config: SqlServerConfig, table: str, chunksize: int, pdf: pd.DataFrame, **_) -> None:
    engine = create_engine(config.sqlalchemy_url())
    try:
        pdf.to_sql(table, engine, if_exists="append", index=False, chunksize=chunksize, method="multi")
    finally:
        engine.dispose()


def method_pandas_fast_executemany(
    *, config: SqlServerConfig, table: str, chunksize: int, pdf: pd.DataFrame, **_
) -> None:
    engine = create_engine(config.sqlalchemy_url(), fast_executemany=True)
    try:
        pdf.to_sql(table, engine, if_exists="append", index=False, chunksize=chunksize)
    finally:
        engine.dispose()


def method_polars_default(*, config: SqlServerConfig, table: str, df: pl.DataFrame, **_) -> None:
    df.write_database(table, config.sqlalchemy_url(), if_table_exists="append", engine="sqlalchemy")


def method_polars_fast_executemany(*, config: SqlServerConfig, table: str, df: pl.DataFrame, **_) -> None:
    engine = create_engine(config.sqlalchemy_url(), fast_executemany=True)
    try:
        with engine.connect() as conn:
            df.write_database(table, conn, if_table_exists="append", engine="sqlalchemy")
            conn.commit()
    finally:
        engine.dispose()


def _pyodbc_executemany(
    *, config: SqlServerConfig, table: str, chunksize: int, df: pl.DataFrame, fast: bool, **_
) -> None:
    conn = pyodbc.connect(config.pyodbc_connection_string())
    cursor = conn.cursor()
    cursor.fast_executemany = fast
    insert_sql = INSERT_SQL_TEMPLATE.format(table=table)
    rows = df.rows()
    try:
        for start in range(0, len(rows), chunksize):
            cursor.executemany(insert_sql, rows[start : start + chunksize])
        conn.commit()
    finally:
        cursor.close()
        conn.close()


def method_pyodbc_default(**kwargs) -> None:
    _pyodbc_executemany(fast=False, **kwargs)


def method_pyodbc_fast_executemany(**kwargs) -> None:
    _pyodbc_executemany(fast=True, **kwargs)


METHODS: dict[str, Callable[..., None]] = {
    "pandas_default": method_pandas_default,
    "pandas_multi": method_pandas_multi,
    "pandas_fast_executemany": method_pandas_fast_executemany,
    "polars_default": method_polars_default,
    "polars_fast_executemany": method_polars_fast_executemany,
    "pyodbc_default": method_pyodbc_default,
    "pyodbc_fast_executemany": method_pyodbc_fast_executemany,
}


def run_benchmark(methods: list[str], rows: int, table: str, chunksize: int) -> list[BenchmarkResult]:
    config = SqlServerConfig.from_env()
    base_engine = create_engine(config.sqlalchemy_url())

    print(f"generating {rows:,} rows...")
    df = generate_dataframe(rows)

    t0 = time.perf_counter()
    pdf = df.to_pandas()
    print(f"polars -> pandas conversion: {time.perf_counter() - t0:.2f}s (参考値、各手法の計測には含めない)")

    results: list[BenchmarkResult] = []
    for name in methods:
        func = METHODS[name]
        print(f"--- {name} ---")
        recreate_table(base_engine, table)
        start = time.perf_counter()
        try:
            func(config=config, table=table, chunksize=chunksize, pdf=pdf, df=df)
            elapsed = time.perf_counter() - start
            results.append(BenchmarkResult(name, rows, elapsed))
            print(f"  {elapsed:.2f}s ({rows / elapsed:,.0f} rows/sec)")
        except Exception as exc:  # noqa: BLE001 - ベンチマークなので1手法の失敗で全体を止めない
            results.append(BenchmarkResult(name, rows, None, error=str(exc)))
            print(f"  failed: {exc}")

    base_engine.dispose()
    return results


def print_summary(results: list[BenchmarkResult]) -> None:
    print("\n=== summary ===")
    ranked = sorted(
        results, key=lambda r: (r.rows_per_sec is None, -(r.rows_per_sec or 0))
    )
    for r in ranked:
        if r.error is not None:
            print(f"{r.method:30s} ERROR: {r.error}")
        else:
            print(f"{r.method:30s} {r.seconds:8.2f}s  {r.rows_per_sec:>12,.0f} rows/sec")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--table", default="dbo.polars_benchmark_upload")
    parser.add_argument("--chunksize", type=int, default=10_000)
    parser.add_argument("--methods", default="all", help="comma区切り。例: pandas_default,polars_fast_executemany")
    parser.add_argument("--list-drivers", action="store_true", help="利用可能なODBCドライバ一覧を表示して終了")
    args = parser.parse_args()

    if args.list_drivers:
        for d in pyodbc.drivers():
            print(d)
        return

    methods = list(METHODS.keys()) if args.methods == "all" else args.methods.split(",")
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise SystemExit(f"unknown methods: {unknown} (choices: {list(METHODS.keys())})")

    results = run_benchmark(methods, args.rows, args.table, args.chunksize)
    print_summary(results)


if __name__ == "__main__":
    main()
