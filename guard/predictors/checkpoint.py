"""Дообученная модель: папка чекпойнта из train.py → вероятности + пороги с val."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from .. import checkpoint as C
from ..models import encode
from ..trainer import free_gpu, predict_proba
from . import Prediction


def predict(path: str | Path, commands: list[str], batch_size: int = 64) -> Prediction:
    model, tok, meta, thresholds, train_commands = C.load(path)
    spec, ml = meta["spec"], int(meta["max_len"])
    ids = encode(tok, commands, spec, ml)
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    proba = predict_proba(model, ids, tok.pad_token_id, batch_size)
    if cuda:
        torch.cuda.synchronize()
    metrics = {"ms_per_cmd_batched": (time.perf_counter() - t0) / len(ids) * 1000,
               "trunc_rate": float(np.mean([len(x) >= ml for x in ids]))}
    if cuda:
        metrics["gpu_peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
    del model
    free_gpu()
    key = meta.get("model_key", spec["key"])
    return Prediction(
        proba=proba, thresholds=thresholds, train_commands=train_commands, metrics=metrics,
        params={"checkpoint": str(path), "model_key": key, "hf_id": spec["hf_id"], "max_len": ml,
                "seed": meta.get("seed"), "train_cats": meta.get("train_cats"), "train_data": meta.get("train_data")},
        tags={"model_key": key, "kind": spec["kind"], "train_cats": str(meta.get("train_cats", "")),
              "train_run_id": str(meta.get("mlflow_run_id", ""))})
