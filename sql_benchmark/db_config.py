"""環境変数からSQL Server接続設定を組み立てる。"""
from __future__ import annotations

import os
import urllib.parse
from dataclasses import dataclass


@dataclass(frozen=True)
class SqlServerConfig:
    server: str
    database: str
    driver: str = "ODBC Driver 18 for SQL Server"
    user: str | None = None
    password: str | None = None
    trusted_connection: bool = True
    encrypt: bool = False

    @classmethod
    def from_env(cls) -> "SqlServerConfig":
        return cls(
            server=os.environ["DB_SERVER"],
            database=os.environ["DB_DATABASE"],
            driver=os.environ.get("DB_DRIVER", "ODBC Driver 18 for SQL Server"),
            user=os.environ.get("DB_USER"),
            password=os.environ.get("DB_PASSWORD"),
            trusted_connection=os.environ.get("DB_TRUSTED_CONNECTION", "1") == "1",
            encrypt=os.environ.get("DB_ENCRYPT", "0") == "1",
        )

    def pyodbc_connection_string(self) -> str:
        parts = [f"DRIVER={{{self.driver}}}", f"SERVER={self.server}", f"DATABASE={self.database}"]
        if self.trusted_connection:
            parts.append("Trusted_Connection=yes")
        else:
            parts.append(f"UID={self.user}")
            parts.append(f"PWD={self.password}")
        parts.append(f"Encrypt={'yes' if self.encrypt else 'no'}")
        parts.append("TrustServerCertificate=yes")
        return ";".join(parts)

    def sqlalchemy_url(self) -> str:
        quoted = urllib.parse.quote_plus(self.pyodbc_connection_string())
        return f"mssql+pyodbc:///?odbc_connect={quoted}"
