"""Hardware, versions, bootstrap CIs, and small helpers."""

from __future__ import annotations

import os
import platform
import subprocess
from typing import Any

import numpy as np


def collect_environment(embed_model_name: str | None = None) -> dict[str, Any]:
    versions: dict[str, str] = {"python": platform.python_version()}
    for mod in (
        "pandas",
        "pyarrow",
        "numpy",
        "sklearn",
        "matplotlib",
        "sentence_transformers",
        "torch",
        "faiss",
        "threadpoolctl",
    ):
        try:
            imported = __import__(mod)
            versions[mod] = getattr(imported, "__version__", "unknown")
        except Exception as exc:  # noqa: BLE001
            versions[mod] = f"NOT_IMPORTABLE: {type(exc).__name__}: {exc}"

    cpu_model = platform.processor() or platform.machine()
    try:
        brand = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
        if brand:
            cpu_model = brand
    except Exception:
        pass

    cores = os.cpu_count()
    ram_gb = None
    try:
        mem_bytes = int(
            subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        )
        ram_gb = round(mem_bytes / (1024**3), 2)
    except Exception:
        pass

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": cpu_model,
        "cpu_cores": cores,
        "ram_gb": ram_gb,
        "library_versions": versions,
        "embed_model_name": embed_model_name,
    }


def bootstrap_mean_ci(
    values: np.ndarray,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "mean": None,
            "ci95_low": None,
            "ci95_high": None,
            "n": 0,
            "note": "no finite values; CI not computed",
        }
    mean = float(arr.mean())
    if arr.size == 1:
        return {
            "mean": mean,
            "ci95_low": mean,
            "ci95_high": mean,
            "n": 1,
            "note": "n=1; CI collapsed to the single observed value",
        }
    rng = np.random.default_rng(seed)
    draws = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "mean": mean,
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "n": int(arr.size),
    }


def percentile(values: np.ndarray, q: float) -> float | None:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(np.percentile(arr, q))


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [json_safe(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
