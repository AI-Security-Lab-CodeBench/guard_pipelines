"""Голые модели без дообучения.

Энкодеры: косинус вектора команды с векторами описаний allow / ask / deny (prompts.CLASS_RULES),
  вероятности = softmax(cos / T). Без обученной головы энкодер иначе не классифицирует.
8B-декодеры: системный промпт с политикой, вероятности = softmax по токенам-меткам первого токена ответа (NF4).
"""
from __future__ import annotations

import time

import numpy as np
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ..data import LABELS
from ..models import collate, encode, load_tokenizer
from ..prompts import CLASS_RULES, SYSTEM_ONE_WORD, USER_ONE_WORD
from ..trainer import free_gpu
from . import Prediction

TEMP = 0.05
CLASS_TEXTS = [f"{lab}: {CLASS_RULES[lab]}" for lab in LABELS]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def predict(spec: dict, commands: list[str]) -> Prediction:
    fn = _decoder if spec["kind"] == "decoder" else _encoder
    t0 = time.perf_counter()
    proba, params, prompt = fn(spec, commands)
    ms = (time.perf_counter() - t0) / len(commands) * 1000
    free_gpu()
    return Prediction(proba=proba, params={"hf_id": spec["hf_id"], **params},
                      tags={"model_key": spec["key"], "kind": spec["kind"]},
                      metrics={"ms_per_cmd_batched": ms}, texts={"prompt.txt": prompt})


@torch.inference_mode()
def _embed(model, tok, spec, texts, pooling, bs=64, max_len=512) -> torch.Tensor:
    out = []
    for i in range(0, len(texts), bs):
        ids, mask = collate(encode(tok, texts[i: i + bs], spec, max_len), tok.pad_token_id, DEVICE)
        h = model(input_ids=ids, attention_mask=mask).last_hidden_state.float()
        if pooling == "cls":
            v = h[:, 0]
        else:
            m = mask.unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1).clamp(min=1)
        out.append(torch.nn.functional.normalize(v, dim=-1))
    return torch.cat(out)


def _encoder(spec, commands):
    tok = load_tokenizer(spec)
    model = AutoModel.from_pretrained(spec["hf_id"]).to(DEVICE).eval()
    pooling = spec.get("zeroshot_pooling", "mean")  # как в 1_Pooling/config.json модели
    c = _embed(model, tok, spec, CLASS_TEXTS, pooling)
    e = _embed(model, tok, spec, commands, pooling)
    proba = torch.softmax((e @ c.T).double() / TEMP, -1).cpu().numpy()
    del model
    return proba, {"method": "embedding_similarity", "pooling": pooling, "temperature": TEMP}, "\n\n".join(CLASS_TEXTS)


def label_token_ids(tok) -> list[list[int]]:
    out = []
    for lab in LABELS:
        ids = set()
        for v in (lab, " " + lab, lab.capitalize(), " " + lab.capitalize()):
            enc = tok.encode(v, add_special_tokens=False)
            if enc:
                ids.add(enc[0])
        out.append(sorted(ids))
    flat = [i for g in out for i in g]
    assert len(flat) == len(set(flat)), f"токены меток пересекаются: {out}"
    return out


@torch.inference_mode()
def _decoder(spec, commands, bs=8):
    tok = AutoTokenizer.from_pretrained(spec["hf_id"])
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        spec["hf_id"], dtype=torch.bfloat16, device_map={"": 0},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
    model.eval()
    lab_ids = label_token_ids(tok)
    prompts = [tok.apply_chat_template([{"role": "system", "content": SYSTEM_ONE_WORD},
                                        {"role": "user", "content": USER_ONE_WORD.format(cmd=c)}],
                                       tokenize=False, add_generation_prompt=True) for c in commands]
    order = np.argsort([len(x) for x in prompts])
    proba = np.zeros((len(commands), len(LABELS)), dtype=np.float64)
    for i in range(0, len(order), bs):
        sel = order[i: i + bs]
        enc = tok([prompts[j] for j in sel], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        lp = torch.log_softmax(model(**enc, logits_to_keep=1).logits[:, -1, :].double(), -1)
        per = torch.stack([torch.logsumexp(lp[:, ids], -1) for ids in lab_ids], -1)
        proba[sel] = torch.softmax(per, -1).cpu().numpy()
    del model
    return (proba, {"method": "label_logits", "quant": "nf4", "label_token_ids": str(lab_ids)},
            f"[system]\n{SYSTEM_ONE_WORD}\n\n[user]\n{USER_ONE_WORD}")
