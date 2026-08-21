"""Train-only encoding and precommitted splits."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class DatasetSplit:
    name: str
    checkpoint: str
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    ids_train: np.ndarray
    ids_val: np.ndarray
    ids_test: np.ndarray
    feature_names: List[str]
    groups_train: Optional[np.ndarray] = None
    groups_val: Optional[np.ndarray] = None
    groups_test: Optional[np.ndarray] = None
    group_names: Optional[Dict[int, str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    split_seed: int = 0


def _row_split(y: np.ndarray, seed: int, test_size: float = 0.2, val_size: float = 0.1):
    idx = np.arange(len(y))
    strat = y if len(np.unique(y)) > 1 else None
    tv, te = train_test_split(idx, test_size=test_size, random_state=seed, stratify=strat)
    y_tv = y[tv]
    strat_tv = y_tv if len(np.unique(y_tv)) > 1 else None
    val_rel = val_size / (1.0 - test_size)
    tr_rel, va_rel = train_test_split(
        np.arange(len(tv)), test_size=val_rel, random_state=seed, stratify=strat_tv
    )
    return tv[tr_rel], tv[va_rel], te


def _student_split(
    student_ids: np.ndarray,
    y: np.ndarray,
    seed: int,
    test_size: float = 0.2,
    val_size: float = 0.1,
):
    students = pd.DataFrame({"sid": student_ids, "y": y}).drop_duplicates("sid")
    strat = students["y"] if students["y"].nunique() > 1 else None
    train_s, test_s = train_test_split(
        students["sid"].values, test_size=test_size, random_state=seed, stratify=strat
    )
    remaining = students[students["sid"].isin(train_s)]
    strat_val = remaining["y"] if remaining["y"].nunique() > 1 else None
    val_rel = val_size / (1.0 - test_size)
    train_s, val_s = train_test_split(
        remaining["sid"].values, test_size=val_rel, random_state=seed, stratify=strat_val
    )
    train_mask = np.isin(student_ids, train_s)
    val_mask = np.isin(student_ids, val_s)
    test_mask = np.isin(student_ids, test_s)
    return (
        np.where(train_mask)[0],
        np.where(val_mask)[0],
        np.where(test_mask)[0],
    )


def encode_and_split(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    label_col: str,
    record_id_col: str,
    split_seed: int,
    split_unit: str,
    groups: Optional[np.ndarray] = None,
    group_names: Optional[Dict[int, str]] = None,
    dataset_name: str = "",
    checkpoint: str = "",
    student_col: Optional[str] = None,
    test_size: float = 0.2,
    val_size: float = 0.1,
) -> DatasetSplit:
    y = df[label_col].astype(int).values
    ids = df[record_id_col].astype(str).values
    if split_unit == "student":
        if student_col is None:
            raise ValueError("student_col required for student-level split")
        train_idx, val_idx, test_idx = _student_split(
            df[student_col].values, y, split_seed, test_size, val_size
        )
    elif split_unit == "row":
        train_idx, val_idx, test_idx = _row_split(y, split_seed, test_size, val_size)
    else:
        raise ValueError(f"Unknown split_unit: {split_unit}")

    X_parts = []
    feature_names: List[str] = []
    for col in feature_cols:
        series = df[col]
        if series.dtype == object or str(series.dtype) in {"category", "string"}:
            train_vals = series.iloc[train_idx].astype(str).fillna("Unknown")
            mapping = {v: i for i, v in enumerate(sorted(train_vals.unique()))}
            mapping.setdefault("Unknown", len(mapping))

            def _map(s: pd.Series) -> np.ndarray:
                return s.astype(str).fillna("Unknown").map(lambda x: mapping.get(x, mapping["Unknown"])).values

            X_parts.append(
                (
                    _map(series.iloc[train_idx]),
                    _map(series.iloc[val_idx]),
                    _map(series.iloc[test_idx]),
                )
            )
            feature_names.append(col)
        else:
            train_num = pd.to_numeric(series.iloc[train_idx], errors="coerce")
            median = float(train_num.median()) if train_num.notna().any() else 0.0

            def _num(idx):
                v = pd.to_numeric(series.iloc[idx], errors="coerce").fillna(median).astype(float).values
                return v

            X_parts.append((_num(train_idx), _num(val_idx), _num(test_idx)))
            feature_names.append(col)

    X_train = np.column_stack([p[0] for p in X_parts]).astype(float)
    X_val = np.column_stack([p[1] for p in X_parts]).astype(float)
    X_test = np.column_stack([p[2] for p in X_parts]).astype(float)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    g_train = g_val = g_test = None
    if groups is not None:
        g_train = np.asarray(groups)[train_idx]
        g_val = np.asarray(groups)[val_idx]
        g_test = np.asarray(groups)[test_idx]

    return DatasetSplit(
        name=dataset_name,
        checkpoint=checkpoint,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y[train_idx],
        y_val=y[val_idx],
        y_test=y[test_idx],
        ids_train=ids[train_idx],
        ids_val=ids[val_idx],
        ids_test=ids[test_idx],
        feature_names=feature_names,
        groups_train=g_train,
        groups_val=g_val,
        groups_test=g_test,
        group_names=group_names,
        metadata={
            "split_unit": split_unit,
            "split_seed": split_seed,
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            "n_features": int(X_train.shape[1]),
        },
        split_seed=split_seed,
    )
