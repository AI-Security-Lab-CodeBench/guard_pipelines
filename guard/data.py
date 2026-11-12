"""Данные: загрузка команд бенчмарка, схемы классов, разбиение train/val, поиск почти-дубликатов.

Формат: CSV с 9 колонками по гайдлайну (command, description, util, severity, category, risk, source, seed, class).
Путь — CSV-файл или папка: берутся commands_<split>.csv в ней и в её подпапках
(для test при отсутствии commands_test.csv — commands.csv, как в ветках feat/data_CAT-*).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

COLUMNS = ["command", "description", "util", "severity", "category", "risk", "source", "seed", "class"]
LABELS = ["allow", "ask", "deny"]
SCHEMES = {
    # v0 строго по гайдлайну: modify / defer выкидываются
    "v0_drop": {"allow": "allow", "ask": "ask", "deny": "deny"},
    # альтернатива: modify ~ deny (с причиной агенту), defer ~ ask
    "v0_merge": {"allow": "allow", "ask": "ask", "deny": "deny", "modify": "deny", "defer": "ask"},
}
SEVERITY_ORDER = ["low", "medium", "high", "critical"]


def find_files(path: str | Path, split: str) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    names = [f"commands_{split}.csv"] + (["commands.csv"] if split == "test" else [])
    out = []
    for d in [path, *sorted(p for p in path.iterdir() if p.is_dir())]:
        f = next((d / n for n in names if (d / n).exists()), None)
        if f is not None:
            out.append(f)
    return out


def load_commands(path: str | Path, split: str = "test", cats: list[str] | None = None,
                  scheme: str = "v0_drop") -> pd.DataFrame:
    """cats — коды категорий (CAT-1) или подкатегорий (CAT-1.2); None — все."""
    files = find_files(path, split)
    if not files:
        raise FileNotFoundError(f"в {path} нет commands_{split}.csv")
    frames = []
    for f in files:
        df = pd.read_csv(f, dtype=str, keep_default_na=False)
        missing = [c for c in COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{f}: нет колонок {missing}")
        df["file"] = f"{f.parent.name}/{f.name}"
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["cat_code"] = df["category"].str.extract(r"^(CAT-\d+)", expand=False)
    df["risk_code"] = df["risk"].str.extract(r"^(CAT-\d+\.\d+)", expand=False)
    df["source_kind"] = df["source"].str.split(":").str[0]
    if cats:
        df = df[df["cat_code"].isin(cats) | df["risk_code"].isin(cats)]
        if df.empty:
            raise ValueError(f"в {path} нет строк категорий {cats}")
    return apply_scheme(df, scheme)


def apply_scheme(df: pd.DataFrame, scheme: str = "v0_drop") -> pd.DataFrame:
    out = df.copy()
    out["label_name"] = out["class"].map(SCHEMES[scheme])
    out = out[out["label_name"].notna()].reset_index(drop=True)
    out["label"] = out["label_name"].map({n: i for i, n in enumerate(LABELS)}).astype(int)
    out["unsafe"] = (out["label_name"] != "allow").astype(int)
    return out


def split_val(train: pd.DataFrame, frac: float = 0.15, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Стратифицированно по подкатегория × класс (редкие пары — по классу)."""
    strat = train["risk_code"].fillna("") + "|" + train["label_name"]
    counts = strat.value_counts()
    strat = strat.where(strat.map(counts) >= 2, train["label_name"])
    tr, va = train_test_split(train, test_size=frac, random_state=seed, stratify=strat)
    return tr.reset_index(drop=True), va.reset_index(drop=True)


def near_dup_scores(test_cmds: list[str], train_cmds: list[str]) -> np.ndarray:
    """Max cosine (char 3-5-gram TF-IDF) каждой тестовой команды к train — для метрик на «чистом» test."""
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(list(train_cmds) + list(test_cmds))
    return (vec.transform(test_cmds) @ vec.transform(train_cmds).T).max(axis=1).toarray().ravel()


def digest(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    for c, lab in sorted(zip(df["command"], df["label_name"])):
        h.update(f"{c}\t{lab}\n".encode("utf-8"))
    return h.hexdigest()[:12]


def class_weights(labels: np.ndarray, n_classes: int = len(LABELS)) -> np.ndarray:
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    w = counts.sum() / (n_classes * np.maximum(counts, 1))
    return w / w.mean()
