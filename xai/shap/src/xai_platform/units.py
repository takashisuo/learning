"""目的変数の単位と、単位付きの数値表記。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Unit:
    """目的変数の単位。

    label:     目的変数そのものの単位(図の軸ラベルに使う)。例: "10万ドル"
    alt_label: 読み手向けの換算単位。None なら換算表記をしない。例: "万ドル"
    alt_scale: 目的変数 1.0 が換算単位でいくつになるか。例: 10.0
    """
    label: str = "目的変数の単位"
    alt_label: str | None = None
    alt_scale: float = 1.0

    @property
    def axis(self) -> str:
        """軸ラベルに埋め込む '単位: ...'。"""
        return f"単位: {self.label}"

    def to_alt(self, v):
        return v * self.alt_scale

    def signed(self, v: float, digits: int = 3) -> str:
        """'+0.820(約+8.2万ドル)' のような符号付き表記。"""
        sign = "+" if v >= 0 else "−"
        s = f"{sign}{abs(v):.{digits}f}"
        if self.alt_label:
            s += f"(約{sign}{abs(self.to_alt(v)):.1f}{self.alt_label})"
        return s

    def level(self, v: float, digits: int = 3) -> str:
        """'2.950(約29.5万ドル)' のような水準の表記。"""
        s = f"{v:.{digits}f}"
        if self.alt_label:
            s += f"(約{self.to_alt(v):.1f}{self.alt_label})"
        return s

    def alt_column(self, name: str) -> str | None:
        """換算列の列名(例: 'SHAP値(万ドル)')。換算しない場合は None。"""
        return f"{name}({self.alt_label})" if self.alt_label else None
