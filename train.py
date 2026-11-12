"""Пайплайн обучения: дообучение классификатора allow / ask / deny → чекпойнт.

  python train.py --model granite_small --data ../benchmark --cats CAT-4 --seeds 42 1337 2024
  python train.py --model seed_coder_8b --data ../benchmark --cats CAT-4

Шаги:
  1. train-сплит (commands_train.csv) нужных категорий;
  2. val: --val-data или стратифицированно 15% от train;
  3. обучение с early stopping по val loss (энкодеры — полностью, 8B — QLoRA);
  4. пороги под FPR 0.5 / 1 / 2 / 5% на val;
  5. чекпойнт: checkpoints/<model>-s<seed>-<категории>/ (weights.pt, meta.json, train_commands.json);
  6. MLflow (эксперимент m2-train): параметры, кривые обучения, метрики на val, meta.json.
Test здесь не трогается — оценка отдельным пайплайном: python evaluate.py --checkpoint <папка> --data ...
"""
from __future__ import annotations

import argparse
import os
import sys
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402

from guard import checkpoint as C  # noqa: E402
from guard import metrics as M  # noqa: E402
from guard import tracking as T  # noqa: E402
from guard.config import DEFAULT_CONFIG, load_specs  # noqa: E402
from guard.data import SCHEMES, digest, load_commands, split_val  # noqa: E402
from guard.models import count_params, encode  # noqa: E402
from guard.trainer import fit, free_gpu, predict_proba, thresholds_from_val  # noqa: E402

VAL_LOG = ("unsafe_roc_auc", "unsafe_pr_auc", "macro_f1_argmax", "fnr_at_fpr_1pct_oracle", "fnr_at_fpr_5pct_oracle")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="ключ из configs/models.yaml")
    ap.add_argument("--data", required=True, help="папка (ищутся commands_train.csv в ней и подпапках) или CSV")
    ap.add_argument("--cats", nargs="*", help="категории / подкатегории для обучения, напр. CAT-4; по умолчанию все")
    ap.add_argument("--val-data", help="отдельный val (папка с commands_val.csv или CSV); иначе отрезается от train")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--val-seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--scheme", default="v0_drop", choices=list(SCHEMES))
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--out", default=str(ROOT / "checkpoints"))
    ap.add_argument("--experiment", default="m2-train")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()

    specs = load_specs(args.config)
    if args.model not in specs:
        sys.exit(f"нет модели {args.model} в {args.config}; есть: {', '.join(specs)}")
    spec = specs[args.model]
    full = load_commands(args.data, "train", args.cats, args.scheme)
    if args.val_data:
        train, val = full, load_commands(args.val_data, "val", args.cats, args.scheme)
    else:
        train, val = split_val(full, args.val_frac, args.val_seed)
    cats = "+".join(sorted(train.cat_code.dropna().unique()))
    print(f"train {len(train)}, val {len(val)} ({cats}); классы train: "
          + ", ".join(f"{k} {v}" for k, v in train.label_name.value_counts().items()), flush=True)
    data_info = {"train_data": str(args.data), "val_data": str(args.val_data or f"split {args.val_frac}"),
                 "train_cats": cats, "scheme": args.scheme, "n_train": len(train), "n_val": len(val),
                 "val_seed": args.val_seed, "digest_train": digest(train), "digest_val": digest(val)}
    if not args.no_mlflow:
        T.setup(args.experiment)

    for seed in args.seeds:
        name = f"{args.model}-s{seed}-{cats}"
        print(f"=== {name}", flush=True)
        with (mlflow.start_run(run_name=name) if not args.no_mlflow else nullcontext()) as run:
            if run:
                mlflow.set_tags({**T.env_tags(), "model_key": args.model, "kind": spec["kind"], "stage": "train",
                                 "scheme": args.scheme, "train_cats": cats})
                mlflow.log_params({**T.clean_params(spec), **data_info, "seed": seed})
            log_fn = (lambda m, step: mlflow.log_metrics(m, step=step)) if run else None
            model, tok, _, info = fit(spec, train, val, seed, log_fn=log_fn)
            pv = predict_proba(model, encode(tok, val.command.tolist(), spec, info["max_len"]), tok.pad_token_id)
            thr = thresholds_from_val(val, pv)
            vm = M.evaluate(val, pv, group_cols=())["metrics"]
            tot, trn = count_params(model)
            ck = C.save(Path(args.out) / name, model, spec, thr, info["max_len"],
                        train_commands=pd.concat([train, val]).command.tolist(),
                        extra={"seed": seed, **data_info, "mlflow_run_id": run.info.run_id if run else None,
                               "epochs_run": info["epochs_run"], "best_val_loss": info["best_val_loss"],
                               "val_metrics": {k: vm[k] for k in VAL_LOG}})
            if run:
                mlflow.log_metrics({**T.clean_metrics(info), "params_total_m": tot / 1e6, "params_trainable_m": trn / 1e6,
                                    **T.clean_metrics({k: vm[k] for k in VAL_LOG}, "val/"),
                                    **{f"val_thr/fpr_{M.pct_key(t)}": v for t, v in thr.items()}})
                mlflow.set_tag("checkpoint", str(ck.resolve()))
                mlflow.log_artifact(str(ck / "meta.json"))
            size = (ck / "weights.pt").stat().st_size / 2**20
            print(f"[{name}] val AUROC {vm['unsafe_roc_auc']:.4f}, macro-F1 {vm['macro_f1_argmax']:.3f}, "
                  f"эпох {info['epochs_run']} → {ck} ({size:.0f} МБ)", flush=True)
            del model, tok
            free_gpu()
    print(f"\nоценка: python evaluate.py --checkpoint {Path(args.out) / f'{args.model}-s{args.seeds[0]}-{cats}'} "
          f"--data <папка с test>")


if __name__ == "__main__":
    main()
