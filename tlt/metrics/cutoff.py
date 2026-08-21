"""Cutoff-distance localization. Mechanism analysis, not a causal effect."""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from tlt.metrics.queue import build_priority_queue

SIGNED_BINS = [-np.inf, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, np.inf]
SIGNED_LABELS = [
    "(-inf,-0.20]",
    "(-0.20,-0.10]",
    "(-0.10,-0.05]",
    "(-0.05,0]",
    "(0,0.05]",
    "(0.05,0.10]",
    "(0.10,0.20]",
    "(0.20,inf)",
]


def cutoff_from_queue(scores: np.ndarray, selected: np.ndarray) -> float:
    if selected.any():
        return float(np.min(scores[selected]))
    return float(np.max(scores))


def localize_transitions(
    scores_ref: np.ndarray,
    scores_alt: np.ndarray,
    record_ids: Sequence,
    y_true: np.ndarray,
    k_frac: float,
) -> pd.DataFrame:
    sel_ref = build_priority_queue(scores_ref, record_ids, k_frac)
    sel_alt = build_priority_queue(scores_alt, record_ids, k_frac)
    c = cutoff_from_queue(scores_ref, sel_ref)
    dist = scores_ref - c
    status = np.full(len(scores_ref), "neither", dtype=object)
    status[sel_ref & sel_alt] = "retained"
    status[sel_ref & ~sel_alt] = "removed"
    status[~sel_ref & sel_alt] = "added"
    bins = pd.cut(dist, bins=SIGNED_BINS, labels=SIGNED_LABELS, include_lowest=True)
    return pd.DataFrame(
        {
            "record_id": np.asarray(record_ids).astype(str),
            "score_ref": scores_ref,
            "score_alt": scores_alt,
            "cutoff_ref": c,
            "distance_to_cutoff": dist,
            "abs_distance": np.abs(dist),
            "bin": bins.astype(str),
            "status": status,
            "y": y_true.astype(int),
            "k_frac": k_frac,
        }
    )


def summarize_localization(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    for bin_name, part in df.groupby("bin", observed=False):
        n = len(part)
        trans = part["status"].isin(["removed", "added"])
        rows.append(
            {
                "bin": str(bin_name),
                "n": n,
                "transition_rate": float(trans.mean()) if n else np.nan,
                "removed_rate": float((part["status"] == "removed").mean()) if n else np.nan,
                "added_rate": float((part["status"] == "added").mean()) if n else np.nan,
                "retained_rate": float((part["status"] == "retained").mean()) if n else np.nan,
                "outcome_rate_all": float(part["y"].mean()) if n else np.nan,
                "outcome_rate_retained": float(part.loc[part["status"] == "retained", "y"].mean())
                if (part["status"] == "retained").any()
                else np.nan,
                "outcome_rate_removed": float(part.loc[part["status"] == "removed", "y"].mean())
                if (part["status"] == "removed").any()
                else np.nan,
                "outcome_rate_added": float(part.loc[part["status"] == "added", "y"].mean())
                if (part["status"] == "added").any()
                else np.nan,
            }
        )
    return pd.DataFrame(rows)
