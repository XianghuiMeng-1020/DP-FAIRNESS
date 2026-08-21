"""Calibration helpers used by the analysis surface."""
from __future__ import annotations

import numpy as np


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    y = np.asarray(y).astype(float)
    p = np.asarray(p).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y)
    if n == 0:
        return float("nan")
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if not np.any(mask):
            continue
        ece += (mask.sum() / n) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(ece)
