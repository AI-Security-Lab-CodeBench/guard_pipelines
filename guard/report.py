"""Отчёт оценки: метрики + таблицы + графики в папку. MLflow-логирование — в evaluate.py поверх этой папки."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M
from . import plots as P

CAT_METRICS = {"AUROC": "auroc", "FNR@FPR1%": "fnr_at_fpr_1pct", "FNR@FPR5%": "fnr_at_fpr_5pct",
               "verdict_FPR": "verdict_fpr", "verdict_FNR": "verdict_fnr",
               "valthr_FPR": "valthr_1pct_fpr", "valthr_FNR": "valthr_1pct_fnr"}
PAIR_CAT_METRICS = {"FNR@FPR1%": "fnr_at_fpr_1pct", "DR@FPR1%": "detection_rate_at_fpr_1pct",
                    "FNR@FPR5%": "fnr_at_fpr_5pct", "DR@FPR5%": "detection_rate_at_fpr_5pct",
                    "verdict_DR": "verdict_detection_rate", "valthr_DR": "valthr_1pct_detection_rate"}
HEAT_COLS = ["AUROC", "FNR@FPR1%", "FNR@FPR5%", "verdict_FPR", "verdict_FNR", "valthr_FPR", "valthr_FNR"]
PAIR_HEAT_COLS = ["AUROC", "DR@FPR1%", "DR@FPR5%", "verdict_DR", "valthr_DR"]


def write_report(name: str, test: pd.DataFrame, proba: np.ndarray, out_dir: str | Path,
                 thresholds: dict[float, float] | None = None, plots: bool = True) -> dict:
    """-> {"metrics": плоский словарь test/…, test_clean/…, cat/<CAT>/…; "tables": {...}}"""
    out = Path(out_dir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    proba = np.asarray(proba, dtype=np.float64)
    ev = M.evaluate(test, proba, thresholds)
    flat = {f"test/{k}": v for k, v in ev["metrics"].items()}
    if "near_dup" in test.columns:
        keep = test["near_dup"].to_numpy() == 0
        flat["test/n_near_dup"] = int((~keep).sum())
        if keep.any():
            ev_c = M.evaluate(test[keep].reset_index(drop=True), proba[keep], thresholds, group_cols=())
            flat.update({f"test_clean/{k}": v for k, v in ev_c["metrics"].items()})
    for c, r in ev["tables"].get("by_cat_code", pd.DataFrame()).iterrows():
        flat.update({f"cat/{c}/{v}": r[k] for k, v in CAT_METRICS.items() if k in r})
    for (c, pair), r in ev["tables"].get("pairs_by_cat_code", pd.DataFrame()).iterrows():
        flat.update({f"cat/{c}/{pair}/{v}": r[k] for k, v in PAIR_CAT_METRICS.items() if k in r})

    (out / "metrics.json").write_text(json.dumps(
        {k: (float(v) if math.isfinite(float(v)) else None) for k, v in flat.items()}, indent=2), encoding="utf-8")
    for tname, t in ev["tables"].items():
        t.to_csv(out / "tables" / f"{tname}.csv", encoding="utf-8")
    thr1 = (thresholds or {}).get(0.01)
    preds = M.predictions_frame(test, proba, thr1)
    preds.to_csv(out / "predictions.csv", index=False, encoding="utf-8")
    col = "pred_valthr_1pct" if thr1 is not None else "pred_argmax"
    preds[preds[col] != preds.label_name].sort_values("unsafe_score").to_csv(out / "errors.csv", index=False,
                                                                             encoding="utf-8")
    if plots:
        _plots(name, test, proba, ev["tables"], out / "plots")
    return {"metrics": flat, "tables": ev["tables"]}


def _plots(name: str, test: pd.DataFrame, proba: np.ndarray, tables: dict, d: Path):
    d.mkdir(exist_ok=True)
    P.save(P.curves(test.label.to_numpy(), test.unsafe.to_numpy(), proba, f"{name}: test"), d / "curves.png")
    P.save(P.confusion(tables["confusion_argmax"], f"{name}: argmax"), d / "confusion_argmax.png")
    if "confusion_valthr_1pct" in tables:
        P.save(P.confusion(tables["confusion_valthr_1pct"], f"{name}: val-порог FPR 1%"), d / "confusion_valthr_1pct.png")
    for col, title in (("cat_code", "по категориям"), ("risk_code", "по подкатегориям")):
        t = tables.get(f"by_{col}")
        if t is not None:
            P.save(P.heatmap(t[[c for c in HEAT_COLS if c in t]], f"{name}: {title}"), d / f"heatmap_{col}.png")
    pc = tables.get("pairs_by_cat_code")
    if pc is not None:
        t = pc[[c for c in PAIR_HEAT_COLS if c in pc]].astype(float)
        t.index = [f"{c} · {p.replace('_vs_', ' vs ')}" for c, p in t.index]
        P.save(P.heatmap(t, f"{name}: detection rate по парам и категориям"), d / "heatmap_pairs_by_cat.png")
    if "recall_risk_x_class" in tables:
        P.save(P.heatmap(tables["recall_risk_x_class"], f"{name}: recall по подкатегории × классу (argmax)"),
               d / "heatmap_risk_x_class.png")
    P.save(P.heatmap(tables["ovo_auc"], f"{name}: one-vs-one AUC (строка a, столбец b)", vmin=0.5), d / "ovo_auc.png")
