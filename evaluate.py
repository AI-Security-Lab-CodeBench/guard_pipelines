"""Пайплайн оценки: модель → вероятности allow / ask / deny на test → метрики, таблицы, графики, MLflow.

Источник предсказаний (ровно один):
  --checkpoint DIR   дообученная модель из train.py; её пороги с val дают честные valthr_*-метрики
  --zeroshot KEY     голая модель без дообучения (ключ из configs/models.yaml)
  --judge KEY        LLM-судья через OpenRouter (OPENROUTER_API_KEY), ответы кэшируются

  python evaluate.py --checkpoint checkpoints/granite_small-s42-CAT-4 --data ../benchmark
  python evaluate.py --zeroshot seed_coder_8b --data ../benchmark --cats CAT-1 CAT-2
  python evaluate.py --judge d1_ultra --data ../benchmark
  python evaluate.py --judge gpt_oss_120b --data ../benchmark --limit 40 --no-mlflow   # пробный прогон

Результат: outputs/<run>/ (metrics.json, tables/*.csv, predictions.csv, errors.csv, plots/*.png, prompt.txt)
и тот же набор в MLflow. Эксперимент по умолчанию: m2-eval-finetuned / m2-eval-zeroshot / m2-eval-judge.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

import numpy as np  # noqa: E402

from guard.config import DEFAULT_CONFIG, load_specs  # noqa: E402
from guard.data import SCHEMES, load_commands, near_dup_scores  # noqa: E402
from guard.predictors.judge import JUDGES  # noqa: E402
from guard.report import write_report  # noqa: E402

SUMMARY = [("AUROC", "unsafe_roc_auc"), ("FNR@FPR1%", "fnr_at_fpr_1pct_oracle"),
           ("FNR@FPR5%", "fnr_at_fpr_5pct_oracle"), ("verdict FPR", "verdict_fpr_argmax"),
           ("verdict FNR", "verdict_fnr_argmax"), ("deny→allow", "deny_to_allow_rate_argmax"),
           ("macro-F1", "macro_f1_argmax"), ("val-порог FPR", "valthr_1pct_fpr"), ("val-порог FNR", "valthr_1pct_fnr")]


def subsample(test, n: int):
    per = max(1, n // test.cat_code.nunique())
    rng = np.random.default_rng(0)
    idx = np.concatenate([rng.permutation(g)[:per] for g in test.groupby("cat_code").indices.values()])
    return test.iloc[np.sort(idx)].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="папка чекпойнта из train.py")
    src.add_argument("--zeroshot", metavar="KEY", help="ключ модели из configs/models.yaml")
    src.add_argument("--judge", choices=list(JUDGES))
    ap.add_argument("--data", required=True, help="папка (commands_test.csv / commands.csv в ней и подпапках) или CSV")
    ap.add_argument("--cats", nargs="*", help="категории / подкатегории, напр. CAT-1 CAT-4; по умолчанию все")
    ap.add_argument("--scheme", default="v0_drop", choices=list(SCHEMES))
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    ap.add_argument("--experiment")
    ap.add_argument("--run-name")
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--cache-dir", default=str(ROOT / "cache" / "judges"), help="кэш ответов судей")
    ap.add_argument("--workers", type=int, default=16, help="параллельные запросы к API")
    ap.add_argument("--limit", type=int, default=0, help="пробный прогон на N командах (поровну по категориям)")
    ap.add_argument("--near-dup-thr", type=float, default=0.9)
    args = ap.parse_args()

    test = load_commands(args.data, "test", args.cats, args.scheme)
    if args.limit:
        test = subsample(test, args.limit)
    print(f"test {len(test)}: " + ", ".join(
        f"{c} {len(g)} (safe {int((g.unsafe == 0).sum())} / unsafe {int(g.unsafe.sum())})"
        for c, g in test.groupby("cat_code")), flush=True)
    cmds = test.command.tolist()

    if args.checkpoint:
        from guard.predictors.checkpoint import predict
        mode, name = "finetuned", Path(args.checkpoint).name
        pred = predict(args.checkpoint, cmds)
    elif args.zeroshot:
        from guard.predictors.zeroshot import predict
        specs = load_specs(args.config)
        if args.zeroshot not in specs:
            sys.exit(f"нет модели {args.zeroshot} в {args.config}; есть: {', '.join(specs)}")
        mode, name = "zeroshot", f"{args.zeroshot}-zeroshot"
        pred = predict(specs[args.zeroshot], cmds)
    else:
        from guard.predictors.judge import predict
        mode, name = "judge", args.judge
        pred = predict(args.judge, cmds, args.cache_dir, args.workers)
    run_name = args.run_name or name + (f"-limit{args.limit}" if args.limit else "")

    if pred.train_commands:
        test["max_cos_to_train"] = near_dup_scores(cmds, pred.train_commands)
        test["near_dup"] = (test["max_cos_to_train"] >= args.near_dup_thr).astype(int)
    out_dir = Path(args.out) / run_name
    rep = write_report(run_name, test, pred.proba, out_dir, pred.thresholds, plots=not args.no_plots)
    for fname, text in pred.texts.items():
        (out_dir / fname).write_text(text, encoding="utf-8")

    m = rep["metrics"]
    print(f"\n[{run_name}] " + ", ".join(f"{lab} {m[f'test/{k}']:.3f}" for lab, k in SUMMARY if f"test/{k}" in m))
    if pred.metrics:
        print("  " + ", ".join(f"{k} {v:.4g}" for k, v in pred.metrics.items()))
    if "test/n_near_dup" in m:
        print(f"  почти-дубликатов train в test: {m['test/n_near_dup']}")
    print(rep["tables"]["pairs"].astype(float).round(3).to_string())
    by_cat = rep["tables"]["by_cat_code"]
    print(by_cat.drop(columns=["n_safe", "n_unsafe"]).round(3).to_string())
    print(f"→ {out_dir}")

    if not args.no_mlflow:
        import mlflow

        from guard import tracking as T

        exp = args.experiment or f"m2-eval-{mode}"
        T.setup(exp)
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tags({**T.env_tags(local_hw=mode != "judge"), "mode": mode, "stage": "eval",
                             "scheme": args.scheme, "test_cats": ",".join(sorted(test.cat_code.dropna().unique())),
                             **pred.tags})
            mlflow.log_params(T.clean_params({**pred.params, "test_data": args.data, "n_test": len(test),
                                              "limit": args.limit or None}))
            mlflow.log_metrics({**T.clean_metrics(m), **T.clean_metrics(pred.metrics)})
            mlflow.log_artifacts(str(out_dir))
        print(f"MLflow: эксперимент {exp}, run {run.info.run_id}")


if __name__ == "__main__":
    main()
