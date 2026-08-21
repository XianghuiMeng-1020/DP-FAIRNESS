"""UCI697 temporal feature policies.

Primary checkpoint: end of Semester 1.
Secondary checkpoint: enrollment / admission only.
Second-semester curricular features are forbidden in both primary analyses.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from tlt.datasets.encode import DatasetSplit, encode_and_split


UCI697_LABEL = "Target"
ENROLLED_CLASS = "Enrolled"
DROPOUT_CLASS = "Dropout"

ENROLLMENT_FEATURES: List[str] = [
    "Marital status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Previous qualification (grade)",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Admission grade",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Scholarship holder",
    "Age at enrollment",
    "International",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]

SEM1_CURRICULAR_FEATURES: List[str] = [
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
]

SEM2_CURRICULAR_FEATURES: List[str] = [
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
]

GROUP_COL = "Gender"
FORBIDDEN_UCI697_FEATURES = set(SEM2_CURRICULAR_FEATURES) | {UCI697_LABEL, "label"}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).replace("\t", "").strip() for c in df.columns]
    return df


def features_for_checkpoint(checkpoint: str) -> List[str]:
    if checkpoint in {"semester1", "sem1", "primary"}:
        return list(ENROLLMENT_FEATURES) + list(SEM1_CURRICULAR_FEATURES)
    if checkpoint in {"enrollment", "enrollment_only", "secondary"}:
        return list(ENROLLMENT_FEATURES)
    raise ValueError(f"Unknown UCI697 checkpoint: {checkpoint}")


def load_uci697_raw(data_dir: str | Path = "data") -> pd.DataFrame:
    root = Path(data_dir)
    path = root / "raw" / "uci697" / "data.csv"
    if not path.exists():
        raise FileNotFoundError(f"UCI697 data.csv not found at {path}")
    df = pd.read_csv(path, sep=";")
    return _normalize_columns(df)


def load_uci697_bundle(
    checkpoint: str = "semester1",
    split_seed: int = 20260820,
    data_dir: str | Path = "data",
    group_attribute: str = GROUP_COL,
) -> DatasetSplit:
    df = load_uci697_raw(data_dir)
    n_raw = len(df)
    df = df[df[UCI697_LABEL] != ENROLLED_CLASS].copy()
    df["label"] = (df[UCI697_LABEL] == DROPOUT_CLASS).astype(int)
    df["record_id"] = [f"uci697_{i}" for i in df.index.astype(int)]
    feature_cols = features_for_checkpoint(checkpoint)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing UCI697 columns: {missing}")
    leaked = [c for c in feature_cols if c in FORBIDDEN_UCI697_FEATURES]
    if leaked:
        raise RuntimeError(f"Temporal leak in UCI697 feature list: {leaked}")
    if any("2nd sem" in c for c in feature_cols):
        raise RuntimeError("Second-semester features leaked into UCI697 bundle")

    groups = df[group_attribute].astype(int).values if group_attribute in df.columns else None
    bundle = encode_and_split(
        df,
        feature_cols=feature_cols,
        label_col="label",
        record_id_col="record_id",
        split_seed=split_seed,
        split_unit="row",
        groups=groups,
        group_names={0: "0", 1: "1"} if groups is not None else None,
        dataset_name="UCI697",
        checkpoint=checkpoint,
    )
    bundle.metadata.update(
        {
            "prediction_unit": "student record",
            "n_raw": n_raw,
            "n_excluded_enrolled": int(n_raw - len(df)),
            "label_definition": "Dropout=1, Graduate=0; Enrolled excluded",
            "checkpoint_definition": (
                "end of Semester 1 (first-semester curricular + enrollment)"
                if checkpoint in {"semester1", "sem1", "primary"}
                else "enrollment/admission variables only"
            ),
            "repeated_records": False,
            "unique_learner_count": int(len(df)),
            "record_count": int(len(df)),
            "forbidden_excluded": sorted(SEM2_CURRICULAR_FEATURES),
        }
    )
    return bundle
