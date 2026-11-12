"""Цикл дообучения (early stopping по val loss), инференс и выбор порогов на val."""
from __future__ import annotations

import gc
import math
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from . import metrics as M
from .data import class_weights
from .models import SeqClassifier, collate, encode, load_tokenizer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def predict_proba(model: SeqClassifier, ids: list[list[int]], pad_id: int, batch_size: int = 64) -> np.ndarray:
    model.eval()
    dev = next(model.parameters()).device
    order = np.argsort([len(x) for x in ids])
    out = np.zeros((len(ids), model.head.out_features), dtype=np.float64)
    for i in range(0, len(ids), batch_size):
        sel = order[i: i + batch_size]
        x, m = collate([ids[j] for j in sel], pad_id, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            logits = model(x, m)
        out[sel] = torch.softmax(logits.double(), -1).cpu().numpy()  # float64: без насыщения у уверенных 8B
    return out


def fit(spec: dict, train: pd.DataFrame, val: pd.DataFrame, seed: int, log_fn=None, device: str = "cuda"):
    """Обучение с early stopping по val loss; возвращает лучшую по val loss модель. -> (model, tok, history, info)."""
    set_seed(seed)
    tok = load_tokenizer(spec)
    model = SeqClassifier(spec)
    if spec["kind"] == "encoder":
        model.to(device)
    max_len = min(int(spec["max_len"]), int(spec["ctx"]))
    tr_ids = encode(tok, train.command.tolist(), spec, max_len)
    va_ids = encode(tok, val.command.tolist(), spec, max_len)
    y_tr = train.label.to_numpy()
    w = torch.tensor(class_weights(y_tr), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=w)

    head_params = list(model.head.parameters())
    head_ids = {id(p) for p in head_params}
    body_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    opt = torch.optim.AdamW([
        {"params": body_params, "lr": float(spec["lr"]), "weight_decay": float(spec["weight_decay"])},
        {"params": head_params, "lr": float(spec["head_lr"]), "weight_decay": 0.0},
    ])
    bs, epochs = int(spec["batch_size"]), int(spec["epochs"])
    steps_total = epochs * math.ceil(len(tr_ids) / bs)
    warm = max(1, int(float(spec["warmup_ratio"]) * steps_total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (steps_total - s) / max(1, steps_total - warm))))

    best, best_loss, bad, history = None, float("inf"), 0, []
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(len(tr_ids))
        tot, nb = 0.0, 0
        for i in range(0, len(perm), bs):
            sel = perm[i: i + bs]
            x, m = collate([tr_ids[j] for j in sel], tok.pad_token_id, device)
            yb = torch.tensor(y_tr[sel], device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x, m)
            loss = loss_fn(logits.float(), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            tot += loss.item()
            nb += 1
        pv = predict_proba(model, va_ids, tok.pad_token_id)
        yv = val.label.to_numpy()
        v_loss = float(nn.functional.cross_entropy(torch.log(torch.tensor(pv, dtype=torch.float64).clamp_min(1e-12)),
                                                   torch.tensor(yv), weight=w.cpu().double()))
        rec = {"epoch": ep, "train_loss": tot / nb, "val_loss": v_loss, "val_macro_f1": M.macro_f1(yv, pv.argmax(1)),
               "val_unsafe_auc": M._safe_auc(val.unsafe.to_numpy(), M.unsafe_score(pv)),
               "elapsed_s": time.time() - t0}
        history.append(rec)
        print(f"  эпоха {ep}: train_loss {rec['train_loss']:.4f}, val_loss {v_loss:.4f}, "
              f"val AUROC {rec['val_unsafe_auc']:.4f}", flush=True)
        if log_fn:
            log_fn({k: v for k, v in rec.items() if k != "epoch"}, ep)
        if v_loss < best_loss - 1e-4:
            best_loss, bad, best = v_loss, 0, model.trainable_state()
        else:
            bad += 1
            if bad >= int(spec["patience"]):
                break
    if best is not None:
        model.load_state_dict(best, strict=False)
    info = {"train_time_s": time.time() - t0, "epochs_run": len(history), "best_val_loss": best_loss,
            "max_len": max_len}
    return model, tok, history, info


def thresholds_from_val(val: pd.DataFrame, pv: np.ndarray) -> dict[float, float]:
    yu, s = val.unsafe.to_numpy(), M.unsafe_score(pv)
    return {t: M.threshold_at_fpr(yu, s, t) for t in M.FPR_TARGETS}


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
