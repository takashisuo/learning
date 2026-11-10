"""
目的: skrub の TableReport を作成し、HTML ファイルとして保存するだけの最小サンプル。
      (レイアウト変更などの加工は行わない、skrub 標準の表示)

使い方:
    python table_report_minimal.py
    -> output/titanic_table_report_minimal.html をブラウザで開いて確認する

動作確認: Python 3.11 / skrub 0.10.1 / pandas 2.2.1 (conda env: samurai)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import skrub

from titanic_loader import load_titanic

OUTPUT_PATH = Path(__file__).parent / "output" / "titanic_table_report_minimal.html"


def save_table_report(df: pd.DataFrame, output_path: Path, title: str | None = None) -> Path:
    """DataFrame から TableReport を作成し、HTML ファイルに保存する。"""
    report = skrub.TableReport(df, title=title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.write_html(output_path)
    return output_path


def main() -> None:
    df = load_titanic()
    path = save_table_report(df, OUTPUT_PATH, title="Titanic")
    print(f"TableReport を保存しました: {path}")


if __name__ == "__main__":
    main()
