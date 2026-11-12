"""LLM-судьи через OpenRouter, без дообучения. Один ключ OPENROUTER_API_KEY для обоих типов API:

  chat      /chat/completions (gpt-oss): ответ `<verdict> <risk 0-100>`, вероятности строятся из оценки риска
            (logprobs вердикта после рассуждения насыщены до 1.0/0.0 и не ранжируют);
  decision  /systemone (Jev, Liquid d1): классификация с вероятностями по классам.

Ответы кэшируются в <cache_dir>/<judge>.jsonl (ключ — sha256 команды): повторный прогон запрашивает только новое,
ключ API нужен только когда есть что запрашивать. Запросы идут к провайдерам без сбора данных (data_collection=deny).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests

from ..data import LABELS
from ..prompts import CLASS_RULES, DECISION_INSTR, DECISION_STATE, SYSTEM_JUDGE, USER_TMPL
from . import Prediction

JUDGES = {
    "gpt_oss_safeguard_20b": {"api": "chat", "model": "openai/gpt-oss-safeguard-20b"},
    "gpt_oss_120b": {"api": "chat", "model": "openai/gpt-oss-120b"},
    "jev_1_14": {"api": "decision", "model": "typesafe/jev-1.14"},
    "d1_ultra": {"api": "decision", "model": "liquid/d1-ultra"},
}
URL = {"chat": "https://openrouter.ai/api/v1/chat/completions", "decision": "https://openrouter.ai/api/v1/systemone"}
KEY_ENV = "OPENROUTER_API_KEY"


class Fatal(Exception):
    pass


def key_of(cmd: str) -> str:
    return hashlib.sha256(cmd.encode("utf-8")).hexdigest()[:20]


def api_key() -> str | None:
    """Из окружения; на Windows — ещё из пользовательских переменных (setx), чтобы не перезапускать терминал."""
    v = os.environ.get(KEY_ENV)
    if not v and sys.platform == "win32":
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                v = winreg.QueryValueEx(k, KEY_ENV)[0]
        except OSError:
            v = None
    return v


def _post(url: str, key: str, body: dict, timeout: float = 120) -> dict:
    headers = {"Authorization": f"Bearer {key}", "X-Title": "code-agent-guard-m2"}
    err = ""
    for attempt in range(7):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=timeout)
        except requests.RequestException as e:
            err = f"network: {e}"
        else:
            if r.status_code == 200:
                return r.json()
            err = f"HTTP {r.status_code}: {r.text[:400]}"
            if r.status_code in (400, 401, 402, 403, 404):
                raise Fatal(err)
        time.sleep(min(60, 2 ** attempt) + random.random())
    raise RuntimeError(err)


def verbal_proba(verdict: str | None, risk: int | None) -> list[float]:
    r = (risk / 100) if risk is not None else (0.02 if verdict == "allow" else 0.98)
    r = min(max(r, 0.005), 0.995)
    if verdict == "ask":
        return [1 - r, 0.9 * r, 0.1 * r]
    if verdict == "deny":
        return [1 - r, 0.1 * r, 0.9 * r]
    return [1 - r, 0.5 * r, 0.5 * r]


def call_chat(model: str, cmd: str, key: str) -> dict:
    body = {"model": model, "temperature": 0, "max_tokens": 4096, "reasoning": {"effort": "low"},
            "provider": {"data_collection": "deny"}, "usage": {"include": True},
            "messages": [{"role": "system", "content": SYSTEM_JUDGE},
                         {"role": "user", "content": USER_TMPL.format(cmd=cmd)}]}
    t0 = time.perf_counter()
    js = _post(URL["chat"], key, body)
    ms = (time.perf_counter() - t0) * 1000
    text = (js["choices"][0]["message"].get("content") or "").strip()
    m = re.search(r"\b(allow|ask|deny)\b\W*(\d{1,3})?", text.lower())
    verdict, risk = (m.group(1), int(m.group(2)) if m.group(2) else None) if m else (None, None)
    return {"text": text, "verdict": verdict, "risk": risk, "p_verbal": verbal_proba(verdict, risk), "ms": ms,
            "usage": js.get("usage", {}), "provider": js.get("provider")}


def call_decision(model: str, cmd: str, key: str) -> dict:
    body = {"model": model, "state": DECISION_STATE.format(cmd=cmd),
            "questions": {"verdict": {"type": "choice", "instructions": DECISION_INSTR, "criteria": CLASS_RULES}},
            "provider": {"data_collection": "deny"}}
    t0 = time.perf_counter()
    js = _post(URL["decision"], key, body)
    ms = (time.perf_counter() - t0) * 1000
    a = js["answers"]["verdict"]
    if a.get("type") == "refusal":
        raise RuntimeError(f"refusal: {a}")
    p = [float(a["probabilities"].get(lab, 0.0)) for lab in LABELS]
    s = sum(p) or 1.0
    return {"verdict": a.get("choice"), "confidence": a.get("confidence"), "p": [x / s for x in p], "ms": ms,
            "usage": js.get("usage", {}), "model": js.get("model")}


def _run(name: str, cmds: list[str], cache_dir: Path, workers: int) -> dict[str, dict]:
    spec = JUDGES[name]
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{name}.jsonl"
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[r["key"]] = r
    todo = sorted({c for c in cmds if key_of(c) not in done})
    print(f"[{name}] в кэше {len(cmds) - len(todo)} из {len(cmds)}, запросить {len(todo)}", flush=True)
    if not todo:
        return done
    key = api_key()
    if not key:
        raise SystemExit(f"нет {KEY_ENV}: задай переменную окружения с ключом OpenRouter")
    call = call_chat if spec["api"] == "chat" else call_decision
    lock, n_err = threading.Lock(), 0
    with path.open("a", encoding="utf-8") as fh, ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(call, spec["model"], c, key): c for c in todo}
        for i, fu in enumerate(as_completed(futs), 1):
            c = futs[fu]
            try:
                r = {"key": key_of(c), **fu.result()}
            except Fatal as e:
                ex.shutdown(cancel_futures=True)
                raise SystemExit(f"[{name}] ошибка API, останавливаюсь: {e}")
            except Exception as e:  # noqa: BLE001
                n_err += 1
                print(f"[{name}] ошибка ({n_err}): {e}", flush=True)
                if n_err > 25:
                    ex.shutdown(cancel_futures=True)
                    raise SystemExit(f"[{name}] слишком много ошибок; кэш сохранён — перезапуск докачает остальное")
                continue
            with lock:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                fh.flush()
                done[r["key"]] = r
            if i % 200 == 0 or i == len(todo):
                print(f"[{name}] {i}/{len(todo)}", flush=True)
    return done


def predict(name: str, commands: list[str], cache_dir: str | Path, workers: int = 16) -> Prediction:
    spec = JUDGES[name]
    uniq = sorted(set(commands))
    res = _run(name, uniq, Path(cache_dir), workers)
    missing = [c for c in uniq if key_of(c) not in res]
    if missing:
        raise SystemExit(f"[{name}] нет ответа для {len(missing)} команд — перезапусти, докачает из кэша")
    field = "p" if spec["api"] == "decision" else "p_verbal"
    proba = np.array([res[key_of(c)][field] for c in commands], dtype=np.float64)
    rows = [res[key_of(c)] for c in uniq]
    ms = np.array([r["ms"] for r in rows])
    u = [r.get("usage") or {} for r in rows]
    metrics = {"api_p50_ms": float(np.median(ms)), "api_p95_ms": float(np.percentile(ms, 95)),
               "tokens_in": sum(x.get("prompt_tokens", x.get("input_tokens", 0)) or 0 for x in u),
               "tokens_out": sum(x.get("completion_tokens", x.get("output_tokens", 0)) or 0 for x in u),
               "cost_usd": sum(float(x.get("cost") or 0) for x in u),
               "n_unparsed": sum(1 for r in rows if r.get("verdict") not in LABELS)}
    if spec["api"] == "chat":
        prompt = f"[system]\n{SYSTEM_JUDGE}\n\n[user]\n{USER_TMPL}"
    else:
        prompt = json.dumps({"state": DECISION_STATE, "instructions": DECISION_INSTR, "criteria": CLASS_RULES},
                            ensure_ascii=False, indent=2)
    hw = "API: OpenRouter" + (" (System One)" if spec["api"] == "decision" else "")
    return Prediction(
        proba=proba, metrics=metrics, texts={"prompt.txt": prompt},
        params={"model_id": spec["model"], "api": spec["api"], "workers": workers,
                "reasoning_effort": "low" if spec["api"] == "chat" else "-"},
        tags={"model_key": name, "kind": "api", "provider": "openrouter", "api": spec["api"],
              "prob_source": "decision_probabilities" if spec["api"] == "decision" else "verbal_risk",
              "hardware": hw, "host": hw})
