"""設定(config/features.yaml)の読込とプロジェクトのパス。"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from .xai_platform import Unit

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "features.yaml"
OUTPUTS = ROOT / "outputs"


@lru_cache
def load_config(path: Path = CONFIG_PATH) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def unit(cfg: dict) -> Unit:
    return Unit(label=cfg["unit"]["label"], digits=cfg["unit"].get("digits", 3))
