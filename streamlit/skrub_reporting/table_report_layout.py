"""
目的: skrub TableReport の Distributions (列ごとの分布カード) の並び方を変更する。

TableReport 標準では分布カードが 1 行に横並び (横スクロール) で表示され、変更するオプションは無い。
レポートは Shadow DOM 内に描画されるため外部 CSS は効かないが、Shadow DOM に複製される
<template> 内にスタイルを追記すれば上書きできる。

    layout = "horizontal" : 標準 (横 1 列・横スクロール)
    layout = "wrap"       : 画面幅で折り返してタイル状に並べる
    layout = "vertical"   : 縦 1 列に並べる

注意: skrub 内部の HTML 構造 (クラス名 column-summary-group) に依存するため、
      skrub のバージョンアップで効かなくなる可能性がある。

動作確認: skrub 0.10.1
"""

from __future__ import annotations

import re
from typing import Literal

Layout = Literal["horizontal", "wrap", "vertical"]

# Table タブで選択列を表示するカード (.columns-in-sample-tab) は対象外にする
_SELECTOR = ".column-summary-group:not(.columns-in-sample-tab)"

_LAYOUT_CSS: dict[str, str] = {
    "horizontal": "",
    "wrap": f"{_SELECTOR} {{ flex-wrap: wrap; width: auto; }}",
    "vertical": f"{_SELECTOR} {{ flex-direction: column; width: auto; }}",
}

_TEMPLATE_TAG = re.compile(r'(<template id="[^"]*-template">)')


def set_distribution_layout(html: str, layout: Layout = "wrap") -> str:
    """TableReport.html() / html_snippet() の結果に、分布カードのレイアウト用 CSS を差し込む。"""
    css = _LAYOUT_CSS[layout]
    if not css:
        return html
    new_html, n = _TEMPLATE_TAG.subn(rf"\1<style>{css}</style>", html, count=1)
    if n == 0:
        raise ValueError("TableReport の <template> が見つかりません (skrub の HTML 構造が変わった可能性があります)")
    return new_html
