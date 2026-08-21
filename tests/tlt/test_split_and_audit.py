"""Split integrity and audit-file presence."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tlt.datasets.oulad_temporal import load_oulad_bundle
from tlt.datasets.uci697_temporal import load_uci697_bundle
from tlt.protocol import load_protocol


def test_uci697_student_ids_do_not_overlap_across_splits():
    proto = load_protocol()
    b = load_uci697_bundle("semester1", split_seed=proto["split_seed"])
    tr, va, te = set(b.ids_train), set(b.ids_val), set(b.ids_test)
    assert tr.isdisjoint(va)
    assert tr.isdisjoint(te)
    assert va.isdisjoint(te)
    assert len(tr) + len(va) + len(te) == b.metadata["record_count"]


def test_uci697_sem1_feature_names_have_no_second_semester():
    proto = load_protocol()
    b = load_uci697_bundle("semester1", split_seed=proto["split_seed"])
    assert all("2nd sem" not in n for n in b.feature_names)
    assert any("1st sem" in n for n in b.feature_names)


def test_oulad_student_ids_do_not_overlap_across_splits():
    proto = load_protocol()
    b = load_oulad_bundle(tau=28, split_seed=proto["split_seed"])
    def students(ids):
        return {str(x).rsplit("_", 1)[-1] for x in ids}

    tr, va, te = students(b.ids_train), students(b.ids_val), students(b.ids_test)
    assert tr.isdisjoint(va)
    assert tr.isdisjoint(te)
    assert va.isdisjoint(te)
    assert all("final_result" not in n and "2nd sem" not in n for n in b.feature_names)


def test_protocol_can_be_frozen(tmp_path):
    from tlt.protocol import freeze_protocol

    out = tmp_path / "TLT_PROTOCOL_FREEZE.json"
    payload = freeze_protocol(out_path=out)
    assert out.exists()
    assert payload["protocol_id"] == "tlt_prioritization_stability_v1"


def test_seed_plan_has_20_replicates():
    from tlt.seeds import build_seed_plan

    df = build_seed_plan()
    assert len(df) == 20
    assert set(df["replicate"]) == set(range(20))
    assert (df["seed_np_a"] != df["seed_np_b"]).all()
    assert (df["seed_np_a"] == df["seed_dp"]).all()
