"""Precommitted seed plan. Never drop a seed for an inconvenient result."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from tlt.protocol import load_protocol, train_seed

ROOT = Path(__file__).resolve().parents[1]


def build_seed_plan() -> pd.DataFrame:
    proto = load_protocol()
    rows: List[Dict] = []
    for r in proto["seeds"]["replicate_ids"]:
        rows.append(
            {
                "replicate": r,
                "split_seed": proto["split_seed"],
                "seed_np_a": train_seed("NP_A", r),
                "seed_np_b": train_seed("NP_B", r),
                "seed_dp": train_seed("DP", r),
                "seed_np_spec": train_seed("NP_SPEC", r),
                "notes": "DP and NP_A share initialization/order seed; NP_B is independent ordinary retraining",
            }
        )
    return pd.DataFrame(rows)


def write_seed_plan(path: Path | None = None) -> Path:
    df = build_seed_plan()
    out = path or (ROOT / "artifacts" / "manifests" / "seed_plan.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out
