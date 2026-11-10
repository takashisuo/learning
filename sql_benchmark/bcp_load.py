"""polarsで生成したデータをCSVへ書き出し、bcpユーティリティでSQL Serverへバルクロードする。

数千万〜数億行規模では、SQLAlchemy/pyodbc経由のINSERT（fast_executemanyでも内部的には
バッチ化されたexecutemanyに過ぎない）よりも、BULK INSERT系（bcp / T-SQL BULK INSERT）の方が
大幅に高速になることが多い。クライアント側にbcp.exeがあれば使えるのはこちらの方式。

前提:
  - bcp.exe がPATHに存在すること（ODBC Driver / SQL Server Native Clientに付属）
  - 認証情報をコマンドライン引数で渡すため、信頼できる実行環境でのみ使用する
    （-U/-Pはプロセス一覧に露出し得るので、可能ならTrusted_Connectionを使う）

実行例:
  python bcp_load.py --rows 1000000 --table dbo.polars_benchmark_upload
"""
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

from sqlalchemy import create_engine, text

from db_config import SqlServerConfig
from sample_data import generate_dataframe

CREATE_TABLE_SQL = """
IF OBJECT_ID('{table}', 'U') IS NOT NULL DROP TABLE {table};
CREATE TABLE {table} (
    id BIGINT,
    category_code INT,
    amount FLOAT,
    status SMALLINT,
    updated_at DATETIME2
);
"""


def create_target_table(config: SqlServerConfig, table: str) -> None:
    engine = create_engine(config.sqlalchemy_url())
    try:
        with engine.begin() as conn:
            conn.execute(text(CREATE_TABLE_SQL.format(table=table)))
    finally:
        engine.dispose()


def export_csv(n_rows: int, csv_path: Path) -> None:
    df = generate_dataframe(n_rows)
    df.write_csv(csv_path, include_header=False)


def run_bcp(config: SqlServerConfig, table: str, csv_path: Path, batch_size: int) -> float:
    cmd = [
        "bcp",
        table,
        "in",
        str(csv_path),
        "-S", config.server,
        "-d", config.database,
        "-c",
        "-t", ",",
        "-r", "\n",
        "-b", str(batch_size),
    ]
    if config.trusted_connection:
        cmd.append("-T")
    else:
        cmd += ["-U", config.user or "", "-P", config.password or ""]

    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        raise RuntimeError(f"bcp failed:\n{result.stdout}\n{result.stderr}")
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--table", default="dbo.polars_benchmark_upload")
    parser.add_argument("--csv-path", default="bcp_upload_tmp.csv")
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--skip-create-table", action="store_true")
    args = parser.parse_args()

    config = SqlServerConfig.from_env()
    csv_path = Path(args.csv_path)

    if not args.skip_create_table:
        create_target_table(config, args.table)

    print(f"generating {args.rows:,} rows -> {csv_path}")
    export_csv(args.rows, csv_path)

    print("running bcp...")
    elapsed = run_bcp(config, args.table, csv_path, args.batch_size)
    print(f"bcp loaded {args.rows:,} rows in {elapsed:.2f}s ({args.rows / elapsed:,.0f} rows/sec)")

    csv_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
