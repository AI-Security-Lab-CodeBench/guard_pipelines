"""MLflow: по умолчанию локальный sqlite-стор guard_pipelines/mlflow.db; общий сервер — через MLFLOW_TRACKING_URI.

UI: mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
"""
from __future__ import annotations

import math
import os
import platform
import subprocess
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import mlflow  # noqa: E402

from . import ROOT  # noqa: E402


def setup(experiment: str) -> str:
    remote = os.environ.get("MLFLOW_TRACKING_URI")
    uri = remote or f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(experiment) is None:
        kw = {} if remote else {"artifact_location": (ROOT / "mlartifacts" / experiment).as_uri()}
        mlflow.create_experiment(experiment, **kw)
    mlflow.set_experiment(experiment)
    return uri


def _cpu_name() -> str:
    try:
        if platform.system() == "Windows":
            import winreg

            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def env_tags(local_hw: bool = True) -> dict:
    """Версии и железо. local_hw=False — для API-моделей (железо провайдера неизвестно)."""
    import transformers

    tags = {"python": platform.python_version(), "transformers": transformers.__version__,
            "os": platform.platform(), "owner": "M2"}
    try:
        tags["git_commit"] = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                                                     stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        tags["git_commit"] = "none"
    if local_hw:
        import psutil
        import torch

        tags.update({"torch": torch.__version__, "cpu": _cpu_name(),
                     "ram_total_gb": f"{psutil.virtual_memory().total / 2**30:.0f}",
                     "host": os.environ.get("M2_HOST", platform.node())})
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            tags.update({"gpu": p.name, "gpu_vram_gb": f"{p.total_memory / 2**30:.0f}", "cuda": str(torch.version.cuda)})
        tags["hardware"] = f"{tags['host']}: {tags['cpu']} / {tags.get('gpu', 'no GPU')}"
    return tags


def clean_metrics(m: dict, prefix: str = "") -> dict:
    """Только конечные числа (MLflow не принимает NaN / None / строки)."""
    out = {}
    for k, v in m.items():
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fv):
            out[f"{prefix}{k}"] = fv
    return out


def clean_params(p: dict) -> dict:
    return {k: v for k, v in p.items() if v is not None and not isinstance(v, (dict, list))}
