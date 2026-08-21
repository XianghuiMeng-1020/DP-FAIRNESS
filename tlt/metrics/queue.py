"""Capacity-constrained prioritization queues and turnover metrics.

The object is a review queue Q_k, not an observed intervention assignment.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import numpy as np


def _as_str_ids(record_ids: Sequence) -> np.ndarray:
    return np.asarray([str(x) for x in record_ids])


def build_priority_queue(
    scores: np.ndarray,
    record_ids: Sequence,
    k_frac: float,
) -> np.ndarray:
    """Return a boolean mask for the top-k review queue.

    Tie-break is deterministic and identical across conditions:
    higher score first; ties broken by lexicographically smaller record_id.
    """
    if not 0.0 < k_frac <= 1.0:
        raise ValueError(f"k_frac must be in (0, 1], got {k_frac}")
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1:
        raise ValueError("scores must be 1-d")
    ids = _as_str_ids(record_ids)
    if len(ids) != len(scores):
        raise ValueError("record_ids and scores must have the same length")
    n = len(scores)
    k = max(1, int(np.ceil(n * k_frac)))
    k = min(k, n)
    order = np.lexsort((ids, -scores))
    selected = np.zeros(n, dtype=bool)
    selected[order[:k]] = True
    return selected


def turnover_fraction(sel_a: np.ndarray, sel_b: np.ndarray) -> float:
    """T = |Q_a \\ Q_b| / |Q_a|."""
    a = np.asarray(sel_a, dtype=bool)
    b = np.asarray(sel_b, dtype=bool)
    k_a = int(a.sum())
    if k_a == 0:
        return 0.0
    return float((a & ~b).sum()) / k_a


def jaccard_index(sel_a: np.ndarray, sel_b: np.ndarray) -> float:
    """J = |intersection| / |union|."""
    a = np.asarray(sel_a, dtype=bool)
    b = np.asarray(sel_b, dtype=bool)
    inter = int((a & b).sum())
    union = int((a | b).sum())
    if union == 0:
        return 1.0
    return inter / union


def excess_prioritization_turnover(
    t_dp: Union[np.ndarray, Sequence[float]],
    t_np: Union[np.ndarray, Sequence[float]],
) -> np.ndarray:
    """EPT(r) = T_DP(r) - T_NP(r)."""
    return np.asarray(t_dp, dtype=float) - np.asarray(t_np, dtype=float)


def queue_utility(y_true: np.ndarray, selected: np.ndarray) -> Dict[str, float]:
    """Precision@k and Outcome-Capture@k for a recorded adverse-outcome label."""
    y = np.asarray(y_true).astype(int)
    sel = np.asarray(selected, dtype=bool)
    at_risk = y == 1
    k = int(sel.sum())
    tp = int((sel & at_risk).sum())
    n_outcome = int(at_risk.sum())
    return {
        "k": k,
        "precision_at_k": (tp / k) if k else 0.0,
        "outcome_capture_at_k": (tp / n_outcome) if n_outcome else 0.0,
        "n_recorded_adverse": n_outcome,
        "tp": tp,
    }


def queue_pair_metrics(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    record_ids: Sequence,
    y_true: np.ndarray,
    k_frac: float,
) -> Dict[str, float]:
    sel_a = build_priority_queue(scores_a, record_ids, k_frac)
    sel_b = build_priority_queue(scores_b, record_ids, k_frac)
    util_a = queue_utility(y_true, sel_a)
    util_b = queue_utility(y_true, sel_b)
    return {
        "k_frac": float(k_frac),
        "k": util_a["k"],
        "turnover": turnover_fraction(sel_a, sel_b),
        "jaccard": jaccard_index(sel_a, sel_b),
        "precision_a": util_a["precision_at_k"],
        "precision_b": util_b["precision_at_k"],
        "delta_precision": util_b["precision_at_k"] - util_a["precision_at_k"],
        "outcome_capture_a": util_a["outcome_capture_at_k"],
        "outcome_capture_b": util_b["outcome_capture_at_k"],
        "delta_outcome_capture": util_b["outcome_capture_at_k"] - util_a["outcome_capture_at_k"],
        "list_size_a": int(sel_a.sum()),
        "list_size_b": int(sel_b.sum()),
    }
