"""出力先ディレクトリと保存、ノートブック表示用の結果コンテナ。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .provenance import MANIFEST, describe, table_source

APP_SOURCE = "アプリ側で作成"  # platform の表 I/F を経由しない出力の表記


@dataclass(frozen=True)
class OutputDirs:
    root: Path

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @property
    def tables(self) -> Path:
        return self.root / "tables"

    def ensure(self) -> "OutputDirs":
        for d in (self.figures, self.tables):
            d.mkdir(parents=True, exist_ok=True)
        return self

    def figure(self, name: str) -> Path:
        return self.figures / name

    def table(self, name: str) -> Path:
        return self.tables / name


def save_csv(df: pd.DataFrame, path: Path, index: bool = True, source: str | None = None) -> Path:
    """Excel でも文字化けしないよう UTF-8(BOM付き)で保存し、生成I/F を MANIFEST に記録する。

    source を省略すると、表を作った platform I/F(@produces_table が付けた名前)を使う。
    アプリ側で組み立てた表は source に使った I/F を明示する。
    """
    df.to_csv(path, index=index, encoding="utf-8-sig")
    src = source or table_source(df) or APP_SOURCE
    MANIFEST.add(path, "表", src, describe(src))
    return path


def save_json(obj, path: Path, source: str = APP_SOURCE) -> Path:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    MANIFEST.add(path, "JSON", source, describe(source))
    return path


@dataclass
class StepResult:
    """分析ステップの結果。ノートブックでは show() でまとめて表示する。"""
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    figures: list[Path] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)

    def show(self) -> None:
        from IPython.display import Image, display

        for t in self.texts:
            print(t)
        for name, df in self.tables.items():
            src = table_source(df)
            print(f"■ {name}" + (f"  [生成I/F: {src}]" if src else ""))
            display(df)
        for p in self.figures:
            rec = MANIFEST.get(p)
            print(f"■ {Path(p).name}" + (f"  [生成I/F: {rec.interface}]" if rec else ""))
            display(Image(str(p)))
