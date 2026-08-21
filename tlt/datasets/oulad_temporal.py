"""OULAD temporal construction at precommitted day checkpoints.

Primary checkpoint: Day 28 after course start.
Sensitivity: Day 14 and Day 56.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from tlt.datasets.encode import DatasetSplit, encode_and_split


STATIC_OULAD_FEATURES: List[str] = [
    "code_module",
    "code_presentation",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "disability",
    "num_of_prev_attempts",
    "studied_credits",
]

FORBIDDEN_OULAD_FEATURES = {
    "final_result",
    "date_unregistration",
    "label",
    "id_student",
}

# Score values are temporally unsafe at Day 14/28/56: OULAD records
# date_submitted but no grade-release timestamp. Submission is not grading.
# Temporal-validity correction (closeout): keep assessment metadata/events only.
SAFE_ASSESSMENT_FEATURES: List[str] = [
    "n_assess_submitted",
    "assess_weight_sum",
]
REMOVED_SCORE_FEATURES: List[str] = [
    "assess_mean_score",
    "n_assess_passed",
]

VLE_ACTIVITY_TYPES = [
    "forumng",
    "oucontent",
    "quiz",
    "resource",
    "homepage",
    "subpage",
    "url",
]

GROUP_COL = "gender"


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.replace("?", np.nan), errors="coerce")


def is_eligible_at_tau(row: Dict, tau: int) -> bool:
    reg = row.get("date_registration")
    unreg = row.get("date_unregistration")
    if reg is None or (isinstance(reg, float) and np.isnan(reg)):
        return False
    if float(reg) > tau:
        return False
    if unreg is None or (isinstance(unreg, float) and np.isnan(unreg)):
        return True
    return float(unreg) > tau


def filter_vle_by_tau(vle: pd.DataFrame, tau: int) -> pd.DataFrame:
    out = vle.copy()
    out["date"] = _to_num(out["date"])
    return out.loc[out["date"] <= tau].copy()


def filter_assessments_by_tau(
    assessments: pd.DataFrame,
    student_assess: pd.DataFrame,
    tau: int,
) -> pd.DataFrame:
    a = assessments.copy()
    sa = student_assess.copy()
    a["date"] = _to_num(a["date"])
    sa["date_submitted"] = _to_num(sa["date_submitted"])
    a = a[a["assessment_type"].astype(str) != "Exam"]
    a = a[a["date"].notna() & (a["date"] <= tau)]
    merged = sa.merge(a, on="id_assessment", how="inner")
    merged = merged[merged["date_submitted"].notna() & (merged["date_submitted"] <= tau)]
    return merged


def _eligible_mask(reg: pd.DataFrame, tau: int) -> pd.Series:
    date_reg = _to_num(reg["date_registration"])
    date_unreg = _to_num(reg["date_unregistration"])
    return date_reg.notna() & (date_reg <= tau) & (date_unreg.isna() | (date_unreg > tau))


def _aggregate_vle(vle_tau: pd.DataFrame, vle_meta: pd.DataFrame) -> pd.DataFrame:
    keys = ["code_module", "code_presentation", "id_student"]
    if vle_tau.empty:
        return pd.DataFrame(columns=keys)
    base = (
        vle_tau.groupby(keys, as_index=False)
        .agg(
            vle_clicks=("sum_click", "sum"),
            vle_events=("sum_click", "size"),
            vle_active_days=("date", "nunique"),
            vle_unique_sites=("id_site", "nunique"),
        )
    )
    merged = vle_tau.merge(vle_meta[["id_site", "activity_type"]], on="id_site", how="left")
    merged["activity_type"] = merged["activity_type"].fillna("other")
    merged["activity_bucket"] = merged["activity_type"].where(
        merged["activity_type"].isin(VLE_ACTIVITY_TYPES), "other"
    )
    act = (
        merged.groupby(keys + ["activity_bucket"], as_index=False)["sum_click"]
        .sum()
        .pivot_table(index=keys, columns="activity_bucket", values="sum_click", fill_value=0)
        .reset_index()
    )
    rename = {t: f"vle_clicks_{t}" for t in list(VLE_ACTIVITY_TYPES) + ["other"]}
    act = act.rename(columns=rename)
    out = base.merge(act, on=keys, how="left")
    return out


def _aggregate_assessments(kept: pd.DataFrame, info_keys: pd.DataFrame) -> pd.DataFrame:
    keys = ["code_module", "code_presentation", "id_student"]
    if kept.empty:
        return pd.DataFrame(columns=keys)
    if "code_module" not in kept.columns:
        # studentAssessment lacks module keys; join via assessments already done in filter
        pass
    kept = kept.copy()
    kept["weight"] = _to_num(kept.get("weight", 0))
    g = kept.groupby(keys, as_index=False).agg(
        n_assess_submitted=("id_assessment", "nunique"),
        assess_weight_sum=("weight", "sum"),
    )
    return g


def build_oulad_table(
    tau: int,
    data_dir: str | Path = "data",
) -> pd.DataFrame:
    root = Path(data_dir) / "raw" / "oulad"
    info = pd.read_csv(root / "studentInfo.csv")
    reg = pd.read_csv(root / "studentRegistration.csv")
    keys = ["code_module", "code_presentation", "id_student"]
    df = info.merge(reg, on=keys, how="left")
    df = df.loc[_eligible_mask(df, tau)].copy()
    df["label"] = df["final_result"].apply(
        lambda x: 1 if str(x).strip().lower() in {"fail", "withdrawn"} else 0
    ).astype(int)
    df["record_id"] = (
        df["code_module"].astype(str)
        + "_"
        + df["code_presentation"].astype(str)
        + "_"
        + df["id_student"].astype(str)
    )

    vle_raw = pd.read_csv(root / "studentVle.csv")
    vle_meta = pd.read_csv(root / "vle.csv")
    vle_tau = filter_vle_by_tau(vle_raw, tau)
    vle_feat = _aggregate_vle(vle_tau, vle_meta)
    df = df.merge(vle_feat, on=keys, how="left")

    assessments = pd.read_csv(root / "assessments.csv")
    student_assess = pd.read_csv(root / "studentAssessment.csv")
    kept = filter_assessments_by_tau(assessments, student_assess, tau)
    assess_feat = _aggregate_assessments(kept, df[keys])
    df = df.merge(assess_feat, on=keys, how="left")

    vle_cols = [
        "vle_clicks",
        "vle_events",
        "vle_active_days",
        "vle_unique_sites",
    ] + [f"vle_clicks_{t}" for t in VLE_ACTIVITY_TYPES + ["other"]]
    assess_cols = list(SAFE_ASSESSMENT_FEATURES)
    for c in vle_cols + assess_cols:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = df[c].fillna(0.0)
    drop_scores = [c for c in REMOVED_SCORE_FEATURES if c in df.columns]
    if drop_scores:
        df = df.drop(columns=drop_scores)

    leaked = [c for c in FORBIDDEN_OULAD_FEATURES if c in vle_cols or c in assess_cols]
    if leaked:
        raise RuntimeError(f"Forbidden OULAD names used as engineered features: {leaked}")
    return df


def feature_columns(df: pd.DataFrame) -> List[str]:
    vle_cols = [
        c
        for c in df.columns
        if c.startswith("vle_")
    ]
    present_unsafe = [c for c in REMOVED_SCORE_FEATURES if c in df.columns]
    if present_unsafe:
        raise RuntimeError(f"Temporally unsafe score features still present: {present_unsafe}")
    assess_cols = [c for c in SAFE_ASSESSMENT_FEATURES if c in df.columns]
    cols = list(STATIC_OULAD_FEATURES) + vle_cols + assess_cols
    for bad in list(FORBIDDEN_OULAD_FEATURES) + list(REMOVED_SCORE_FEATURES):
        if bad in cols:
            raise RuntimeError(f"Forbidden or temporally unsafe feature present: {bad}")
    return cols


def load_oulad_bundle(
    tau: int = 28,
    split_seed: int = 20260820,
    data_dir: str | Path = "data",
    cache_dir: Optional[str | Path] = "artifacts/data",
) -> DatasetSplit:
    cache_path = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"oulad_tau{tau}_table.parquet"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path is not None and cache_path.exists():
        df = pd.read_parquet(cache_path)
        stale = [c for c in REMOVED_SCORE_FEATURES if c in df.columns]
        if stale:
            df = df.drop(columns=stale)
            df.to_parquet(cache_path, index=False)
    else:
        df = build_oulad_table(tau, data_dir=data_dir)
        if cache_path is not None:
            df.to_parquet(cache_path, index=False)

    feature_cols = feature_columns(df)
    groups = None
    group_names = None
    if GROUP_COL in df.columns:
        mapping = {v: i for i, v in enumerate(sorted(df[GROUP_COL].astype(str).fillna("Unknown").unique()))}
        groups = df[GROUP_COL].astype(str).fillna("Unknown").map(mapping).values
        group_names = {i: k for k, i in mapping.items()}

    bundle = encode_and_split(
        df,
        feature_cols=feature_cols,
        label_col="label",
        record_id_col="record_id",
        split_seed=split_seed,
        split_unit="student",
        student_col="id_student",
        groups=groups,
        group_names=group_names,
        dataset_name="OULAD",
        checkpoint=f"day{tau}",
    )
    bundle.metadata.update(
        {
            "prediction_unit": "student-course registration",
            "unique_learner_count": int(df["id_student"].nunique()),
            "record_count": int(len(df)),
            "repeated_records": bool((df.groupby("id_student").size() > 1).any()),
            "n_students_with_repeat_regs": int((df.groupby("id_student").size() > 1).sum()),
            "label_definition": "recorded Fail/Withdrawn=1; Pass/Distinction=0",
            "checkpoint_definition": (
                f"Day {tau} after course start; VLE date<=tau; "
                "assessments due and submitted by tau; Exam excluded; "
                "assessment score values excluded (grade-release time unverifiable)"
            ),
            "assessment_feature_policy": "metadata_events_only_no_scores",
            "removed_score_features": list(REMOVED_SCORE_FEATURES),
            "eligible_rule": "date_registration<=tau and (date_unregistration missing or > tau)",
            "tau": int(tau),
            "forbidden_excluded": sorted(FORBIDDEN_OULAD_FEATURES),
        }
    )
    return bundle
