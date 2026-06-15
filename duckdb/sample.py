import os, sys, duckdb
from pathlib import Path

path = Path("C:/d/takashi/python/github/learning_v2/samurai/openpyxl_cource/ホームセンター課題/sales.csv")

conn = duckdb.connect()
conn.execute(open('./sample.sql').read())