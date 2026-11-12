"""Источники предсказаний для пайплайна оценки. Каждый возвращает Prediction с вероятностями allow / ask / deny.

  checkpoint.predict(path, commands)        дообученная модель из train.py
  zeroshot.predict(spec, commands)          голая модель без дообучения
  judge.predict(name, commands, cache_dir)  LLM-судья через OpenRouter
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Prediction:
    proba: np.ndarray                                   # (n, 3) float64, порядок классов data.LABELS
    params: dict = field(default_factory=dict)          # → MLflow params
    tags: dict = field(default_factory=dict)            # → MLflow tags
    metrics: dict = field(default_factory=dict)         # задержка, стоимость и т.п. → MLflow metrics
    texts: dict = field(default_factory=dict)           # имя файла → текст (промпт), кладётся в папку отчёта
    thresholds: dict[float, float] | None = None        # пороги с val (только у чекпойнтов)
    train_commands: list[str] | None = None             # для поиска почти-дубликатов в test
