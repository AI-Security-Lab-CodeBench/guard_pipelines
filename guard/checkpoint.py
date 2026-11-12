"""Формат чекпойнта — папка, которая связывает пайплайны обучения и оценки:

  weights.pt           обучаемые параметры (энкодеры — все, 8B — LoRA + голова)
  meta.json            spec модели, max_len, пороги под FPR 0.5/1/2/5% с val, данные обучения, seed
  train_commands.json  команды train + val — для поиска почти-дубликатов в test (метрики test_clean/)

Чекпойнты из m2_eval/checkpoints (без порогов и train_commands.json) тоже читаются.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .models import SeqClassifier, load_tokenizer


def save(path: str | Path, model: SeqClassifier, spec: dict, thresholds: dict[float, float], max_len: int,
         train_commands: list[str], extra: dict) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.trainable_state(), path / "weights.pt")
    meta = {"model_key": spec["key"], "hf_id": spec["hf_id"], "max_len": int(max_len),
            "thresholds": {str(k): float(v) for k, v in thresholds.items()}, **extra,
            "spec": {k: v for k, v in spec.items() if isinstance(v, (str, int, float, bool, list, dict))}}
    (path / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (path / "train_commands.json").write_text(json.dumps(list(train_commands), ensure_ascii=False), encoding="utf-8")
    return path


def load(path: str | Path, device: str | None = None):
    """-> (model, tok, meta, thresholds | None, train_commands | None)"""
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    spec = meta["spec"]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SeqClassifier(spec)
    if spec["kind"] == "encoder":
        model.to(device)
    state = torch.load(path / "weights.pt", map_location="cpu", weights_only=True)
    res = model.load_state_dict(state, strict=False)
    lost = {k for k, p in model.named_parameters() if p.requires_grad} - set(state)
    if res.unexpected_keys or lost:
        raise RuntimeError(f"{path}: чекпойнт не совпадает с моделью; лишние {res.unexpected_keys[:3]}, "
                           f"нет {sorted(lost)[:3]}")
    model.eval()
    thresholds = {float(k): float(v) for k, v in (meta.get("thresholds") or {}).items()} or None
    tc = path / "train_commands.json"
    train_commands = json.loads(tc.read_text(encoding="utf-8")) if tc.exists() else None
    return model, load_tokenizer(spec), meta, thresholds, train_commands
