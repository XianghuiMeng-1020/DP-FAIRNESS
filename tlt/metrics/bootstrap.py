"""Replicate-level bootstrap summaries. Inference unit is the training run."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def percentile_bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 20260820,
) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "iqr": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "n_boot": n_boot,
        }
    rng = np.random.RandomState(seed)
    if n == 1:
        means = np.full(n_boot, arr[0])
    else:
        idx = rng.randint(0, n, size=(n_boot, n))
        means = arr[idx].mean(axis=1)
    q25, q75 = np.percentile(arr, [25, 75])
    return {
        "n": int(n),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "ci95_low": float(np.percentile(means, 100 * alpha / 2)),
        "ci95_high": float(np.percentile(means, 100 * (1 - alpha / 2))),
        "n_boot": int(n_boot),
    }
