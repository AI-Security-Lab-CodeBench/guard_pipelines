"""Конфиг моделей: configs/models.yaml (defaults + параметры каждой модели)."""
from __future__ import annotations

from pathlib import Path

import yaml

from . import ROOT

DEFAULT_CONFIG = ROOT / "configs" / "models.yaml"


def load_specs(path: str | Path | None = None) -> dict[str, dict]:
    cfg = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text(encoding="utf-8"))
    return {key: {**cfg["defaults"], **spec, "key": key} for key, spec in cfg["models"].items()}
