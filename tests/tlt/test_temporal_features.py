"""TDD: temporal feature policies for OULAD and UCI697."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tlt.datasets.uci697_temporal import (
    ENROLLMENT_FEATURES,
    SEM1_CURRICULAR_FEATURES,
    SEM2_CURRICULAR_FEATURES,
    features_for_checkpoint,
)
from tlt.datasets.oulad_temporal import (
    FORBIDDEN_OULAD_FEATURES,
    REMOVED_SCORE_FEATURES,
    SAFE_ASSESSMENT_FEATURES,
    STATIC_OULAD_FEATURES,
    feature_columns,
    filter_assessments_by_tau,
    filter_vle_by_tau,
    is_eligible_at_tau,
)


def test_uci697_sem1_excludes_all_second_semester_features():
    cols = features_for_checkpoint("semester1")
    for name in SEM2_CURRICULAR_FEATURES:
        assert name not in cols, name
    for name in SEM1_CURRICULAR_FEATURES:
        assert name in cols, name
    for name in ENROLLMENT_FEATURES:
        assert name in cols, name


def test_uci697_enrollment_only_excludes_all_curricular_performance():
    cols = features_for_checkpoint("enrollment")
    for name in SEM1_CURRICULAR_FEATURES + SEM2_CURRICULAR_FEATURES:
        assert name not in cols, name
    assert "Admission grade" in cols
    assert "Age at enrollment" in cols


def test_oulad_forbidden_features_never_used():
    for name in ("final_result", "date_unregistration", "label"):
        assert name in FORBIDDEN_OULAD_FEATURES
        assert name not in STATIC_OULAD_FEATURES


def test_vle_filter_drops_post_checkpoint_events():
    vle = pd.DataFrame(
        {
            "code_module": ["AAA", "AAA", "AAA"],
            "code_presentation": ["2013J", "2013J", "2013J"],
            "id_student": [1, 1, 2],
            "id_site": [10, 11, 10],
            "date": [10, 40, 28],
            "sum_click": [3, 9, 1],
        }
    )
    kept = filter_vle_by_tau(vle, tau=28)
    assert set(kept["date"]) == {10, 28}
    assert 40 not in set(kept["date"])


def test_assessment_filter_requires_due_and_submission_by_tau_and_drops_exams():
    assessments = pd.DataFrame(
        {
            "id_assessment": [1, 2, 3, 4],
            "assessment_type": ["TMA", "TMA", "Exam", "CMA"],
            "date": [20, 40, 20, 15],
            "weight": [10, 10, 100, 5],
        }
    )
    student_assess = pd.DataFrame(
        {
            "id_assessment": [1, 2, 3, 4],
            "id_student": [1, 1, 1, 1],
            "date_submitted": [18, 19, 10, 40],
            "score": [80, 70, 50, 90],
            "is_banked": [0, 0, 0, 0],
        }
    )
    kept = filter_assessments_by_tau(assessments, student_assess, tau=28)
    assert set(kept["id_assessment"]) == {1}


def test_eligibility_excludes_early_withdrawal_and_late_registration():
    row_ok = {"date_registration": -10, "date_unregistration": 40}
    row_left = {"date_registration": -10, "date_unregistration": 10}
    row_late = {"date_registration": 40, "date_unregistration": None}
    row_missing_reg = {"date_registration": None, "date_unregistration": None}
    assert is_eligible_at_tau(row_ok, tau=28) is True
    assert is_eligible_at_tau(row_left, tau=28) is False
    assert is_eligible_at_tau(row_late, tau=28) is False
    assert is_eligible_at_tau(row_missing_reg, tau=28) is False


def test_oulad_score_features_are_excluded_from_model_inputs():
    assert "assess_mean_score" in REMOVED_SCORE_FEATURES
    assert "n_assess_passed" in REMOVED_SCORE_FEATURES
    assert "assess_mean_score" not in SAFE_ASSESSMENT_FEATURES
    assert "n_assess_passed" not in SAFE_ASSESSMENT_FEATURES
    df = pd.DataFrame(
        {
            "code_module": ["AAA"],
            "code_presentation": ["2013J"],
            "region": ["X"],
            "highest_education": ["A"],
            "imd_band": ["0-10%"],
            "age_band": ["0-35"],
            "disability": ["N"],
            "num_of_prev_attempts": [0],
            "studied_credits": [60],
            "vle_clicks": [1],
            "n_assess_submitted": [1],
            "assess_weight_sum": [10],
            "assess_mean_score": [80],
            "n_assess_passed": [1],
        }
    )
    try:
        feature_columns(df)
        raised = False
    except RuntimeError:
        raised = True
    assert raised
    df2 = df.drop(columns=list(REMOVED_SCORE_FEATURES))
    cols = feature_columns(df2)
    for name in REMOVED_SCORE_FEATURES:
        assert name not in cols
