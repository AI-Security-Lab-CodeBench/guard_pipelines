"""Графики оценки (PNG): ROC / FNR-vs-FPR / PR, матрицы ошибок, тепловые карты по группам."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from .data import LABELS  # noqa: E402
from .metrics import unsafe_score  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CLASS_COLORS = {"allow": "#2a78d6", "ask": "#eb6834", "deny": "#1baf7a"}
UNSAFE_COLOR = "#0b0b0b"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
                                                     "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "legend.frameon": False, "lines.linewidth": 1.6,
})


def curves(y: np.ndarray, yu: np.ndarray, proba: np.ndarray, title: str):
    s = unsafe_score(proba)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
    for i, n in enumerate(LABELS):
        yi = (y == i).astype(int)
        if 0 < yi.sum() < len(yi):
            f, t, _ = roc_curve(yi, proba[:, i])
            ax[0].plot(f, t, color=CLASS_COLORS[n], label=f"{n} vs rest")
            p, r, _ = precision_recall_curve(yi, proba[:, i])
            ax[2].plot(r, p, color=CLASS_COLORS[n], label=f"{n} vs rest")
    f, t, _ = roc_curve(yu, s)
    ax[0].plot(f, t, color=UNSAFE_COLOR, ls="--", label="unsafe (ask + deny)")
    ax[0].plot([0, 1], [0, 1], color=GRID, lw=1)
    ax[0].set(title="ROC, one-vs-rest", xlabel="FPR", ylabel="TPR")
    # рабочая зона guard'а: низкий FPR в лог-шкале; allow vs ask / allow vs deny — тот же score, те же allow
    allow = y == LABELS.index("allow")
    for n in ("ask", "deny"):
        sel = allow | (y == LABELS.index(n))
        if (y == LABELS.index(n)).any() and allow.any():
            fp, tp, _ = roc_curve((y[sel] != LABELS.index("allow")).astype(int), s[sel])
            ax[1].plot(np.clip(fp, 1e-4, 1), 1 - tp, color=CLASS_COLORS[n], label=f"allow vs {n}")
    ax[1].plot(np.clip(f, 1e-4, 1), 1 - t, color=UNSAFE_COLOR, ls="--", label="allow vs all")
    ax[1].legend(loc="upper right", fontsize=8)
    ax[1].set_xscale("log")
    ax[1].axvline(0.01, color="#e34948", lw=1)
    ax[1].axhline(0.05, color="#e34948", lw=1, ls=":")
    ax[1].text(0.0105, 0.9, "FPR 1%", color=INK2, fontsize=8)
    ax[1].text(1.2e-4, 0.065, "FNR 5%", color=INK2, fontsize=8)
    ax[1].set(title="unsafe: FNR vs FPR (лог)", xlabel="FPR", ylabel="FNR", ylim=(0, 1), xlim=(1e-4, 1))
    p, r, _ = precision_recall_curve(yu, s)
    ax[2].plot(r, p, color=UNSAFE_COLOR, ls="--", label="unsafe (ask + deny)")
    ax[2].set(title="Precision-Recall", xlabel="recall", ylabel="precision", ylim=(0, 1.02))
    ax[0].legend(loc="lower right", fontsize=8)
    ax[2].legend(loc="lower left", fontsize=8)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def heatmap(df: pd.DataFrame, title: str, fmt: str = "{:.2f}", vmin: float | None = 0, vmax: float | None = 1):
    data = df.astype(float).to_numpy()
    h = max(2.2, 0.38 * len(df) + 1.2)
    w = max(4.0, 0.95 * df.shape[1] + 2.4, 0.075 * len(title) + 1.2)  # заголовок не должен обрезаться
    fig, ax = plt.subplots(figsize=(w, h))
    im = ax.imshow(data, cmap=SEQ, aspect="auto", vmin=vmin, vmax=vmax)
    ax.grid(False)
    ax.set_xticks(range(df.shape[1]), [str(c) for c in df.columns], rotation=30, ha="right")
    ax.set_yticks(range(len(df)), [str(i) for i in df.index])
    lo = np.nanmin(data) if vmin is None else vmin
    hi = np.nanmax(data) if vmax is None else vmax
    for (r, c), v in np.ndenumerate(data):
        if not np.isnan(v):
            frac = (v - lo) / (hi - lo + 1e-12)
            ax.text(c, r, fmt.format(v), ha="center", va="center", fontsize=8, color="white" if frac > 0.55 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.outline.set_visible(False)
    ax.set_title(title, loc="left")
    fig.tight_layout()
    return fig


def confusion(cm: pd.DataFrame, title: str):
    norm = cm.div(cm.sum(1).replace(0, np.nan), axis=0)
    fig = heatmap(norm, title + " (строки = истина, доля)")
    ax = fig.axes[0]
    for t in list(ax.texts):
        t.remove()
    for (r, c), v in np.ndenumerate(norm.to_numpy()):
        if not np.isnan(v):
            ax.text(c, r, f"{v:.2f}\n({cm.iat[r, c]})", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.55 else INK)
    ax.set_xlabel("предсказание")
    ax.set_ylabel("истина")
    fig.tight_layout()
    return fig


def save(fig, path):
    fig.savefig(path, dpi=130)
    plt.close(fig)
