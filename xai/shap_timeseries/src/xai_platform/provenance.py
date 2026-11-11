"""出力ファイルの来歴(どの I/F で作成したか)の記録。

- 表を返す I/F には @produces_table を付ける。戻り値の DataFrame/Series の attrs["生成I/F"] に
  I/F 名を入れ、io.save_csv が保存時にそれを読んで MANIFEST に登録する。
- 図を保存する I/F には @produces_figure を付ける。戻り値の保存先 path を MANIFEST に登録する。
- 呼び出し側(アプリ)は MANIFEST.step(...) で「どの分析ステップか」を設定し、
  最後に MANIFEST.save(...) で一覧(CSV / Markdown)を書き出す。
"""
from __future__ import annotations

import functools
import inspect
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ATTR = "生成I/F"


def interface_name(func) -> str:
    """'xai_platform.explain.decomposition_table' のような I/F 名。"""
    module = func.__module__.split(".", 1)[-1] if func.__module__.startswith("src.") else func.__module__
    return f"{module}.{func.__qualname__}"


def _summary(func) -> str:
    doc = inspect.getdoc(func) or ""
    return doc.splitlines()[0] if doc else ""


@dataclass
class Record:
    path: Path
    kind: str           # "図" / "表" / "JSON"
    interface: str      # 生成した I/F
    summary: str        # I/F の説明(docstring 1行目)
    shap_api: str       # 内部で使った shap の描画関数(あれば)
    step: str           # 呼び出し側の分析ステップ


class Manifest:
    def __init__(self):
        self._records: dict[Path, Record] = {}
        self._step = ""

    @contextmanager
    def step(self, name: str):
        prev, self._step = self._step, name
        try:
            yield
        finally:
            self._step = prev

    def add(self, path: Path, kind: str, interface: str, summary: str = "",
            shap_api: str = "") -> None:
        path = Path(path).resolve()
        self._records.pop(path, None)  # 再実行時は最新で上書き(順序も更新)
        self._records[path] = Record(path, kind, interface, summary, shap_api, self._step)

    def get(self, path: Path) -> Record | None:
        return self._records.get(Path(path).resolve())

    def reset(self) -> None:
        self._records.clear()

    def to_frame(self, root: Path) -> pd.DataFrame:
        root = Path(root).resolve()
        return pd.DataFrame([{
            "ファイル": r.path.relative_to(root).as_posix() if r.path.is_relative_to(root) else str(r.path),
            "種類": r.kind, "生成I/F": r.interface, "I/Fの説明": r.summary,
            "SHAP描画関数": r.shap_api, "分析ステップ": r.step} for r in self._records.values()])

    def save(self, root: Path, title: str = "出力ファイルと生成I/Fの対応") -> tuple[Path, Path]:
        """root/manifest.csv と root/manifest.md を書き出す。"""
        df = self.to_frame(root)
        csv = Path(root) / "manifest.csv"
        df.to_csv(csv, index=False, encoding="utf-8-sig")
        lines = [f"# {title}", "",
                 "このファイルは自動生成です(`xai_platform.provenance`)。",
                 "「生成I/F」はプラットフォーム層の関数、「分析ステップ」はそれを呼び出したアプリ層の処理です。",
                 "「SHAP描画関数」が空欄の図は、SHAP値をもとに matplotlib で自作した図です。", ""]
        def code(s: str) -> str:
            return f"`{s}`" if s else ""

        def cell(s: str) -> str:
            return s.replace("|", "\\|")

        for kind, g in df.groupby("種類", sort=False):
            lines += [f"## {kind}", "", "| ファイル | 生成I/F | I/Fの説明 | SHAP描画関数 | 分析ステップ |",
                      "|---|---|---|---|---|"]
            for _, r in g.iterrows():
                iface = r["生成I/F"]
                iface = code(iface) if "." in iface and " " not in iface else cell(iface)
                lines.append(f"| {code(r['ファイル'])} | {iface} | {cell(r['I/Fの説明'])} | "
                             f"{code(r['SHAP描画関数'])} | {r['分析ステップ']} |")
            lines.append("")
        md = Path(root) / "manifest.md"
        md.write_text("\n".join(lines), encoding="utf-8")
        return csv, md


MANIFEST = Manifest()  # 既定の記録先


def _tag(obj, name: str):
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        obj.attrs[ATTR] = name
    elif isinstance(obj, tuple):
        for o in obj:
            _tag(o, name)
    return obj


def produces_table(func):
    """戻り値の表に生成I/F名を付ける(保存時に io.save_csv が記録する)。"""
    name = interface_name(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return _tag(func(*args, **kwargs), name)

    return wrapper


def produces_figure(shap_api: str = ""):
    """図を保存する I/F 用。戻り値の path を MANIFEST に登録する。"""
    def deco(func):
        name, summary = interface_name(func), _summary(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            path = func(*args, **kwargs)
            MANIFEST.add(path, "図", name, summary, shap_api)
            return path

        return wrapper
    return deco


def table_source(obj) -> str | None:
    return getattr(obj, "attrs", {}).get(ATTR)


def describe(interface: str) -> str:
    """I/F 名から docstring 1行目を引く(表の説明欄用)。"""
    import importlib

    module, _, attr = interface.rpartition(".")
    if not module:  # 「アプリ側で作成」など I/F 名でないもの
        return ""
    for name in (f"src.{module}", module):  # interface_name は "src." を外しているので先に補って探す
        try:
            return _summary(getattr(importlib.import_module(name), attr))
        except (ImportError, AttributeError):
            continue
    return ""
