"""TDD: capacity-constrained prioritization queue, turnover, EPT."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tlt.metrics.queue import (
    build_priority_queue,
    excess_prioritization_turnover,
    jaccard_index,
    queue_utility,
    turnover_fraction,
)


def test_queue_size_is_ceil_k_frac():
    scores = np.array([0.9, 0.8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.05])
    ids = np.arange(len(scores))
    selected = build_priority_queue(scores, record_ids=ids, k_frac=0.10)
    assert selected.sum() == 1
    selected20 = build_priority_queue(scores, record_ids=ids, k_frac=0.20)
    assert selected20.sum() == 2


def test_tie_break_is_deterministic_and_id_stable():
    scores = np.array([0.5, 0.5, 0.5, 0.1])
    ids_a = np.array(["c", "a", "b", "d"])
    ids_b = np.array(["c", "a", "b", "d"])
    q1 = build_priority_queue(scores, record_ids=ids_a, k_frac=0.25)
    q2 = build_priority_queue(scores, record_ids=ids_b, k_frac=0.25)
    assert np.array_equal(q1, q2)
    # lowest record_id among ties wins when scores are equal
    assert q1[1]  # "a"
    assert not q1[0]
    assert not q1[2]


def test_turnover_and_jaccard_hand_example():
    sel_a = np.array([True, True, False, False])
    sel_b = np.array([True, False, True, False])
    assert abs(turnover_fraction(sel_a, sel_b) - 0.5) < 1e-12
    assert abs(jaccard_index(sel_a, sel_b) - (1 / 3)) < 1e-12


def test_ept_is_paired_difference():
    t_dp = np.array([0.40, 0.30, 0.50])
    t_np = np.array([0.20, 0.30, 0.10])
    ept = excess_prioritization_turnover(t_dp, t_np)
    assert np.allclose(ept, np.array([0.20, 0.00, 0.40]))


def test_queue_utility_precision_and_outcome_capture():
    y = np.array([1, 1, 0, 0, 1, 0])
    selected = np.array([True, False, True, False, False, False])
    util = queue_utility(y, selected)
    assert util["k"] == 2
    assert abs(util["precision_at_k"] - 0.5) < 1e-12
    assert abs(util["outcome_capture_at_k"] - (1 / 3)) < 1e-12


def test_identical_scores_and_ids_yield_identical_queues():
    rng = np.random.RandomState(0)
    scores = rng.rand(50)
    ids = np.array([f"r{i:03d}" for i in range(50)])
    q1 = build_priority_queue(scores, record_ids=ids, k_frac=0.20)
    q2 = build_priority_queue(scores.copy(), record_ids=ids.copy(), k_frac=0.20)
    assert np.array_equal(q1, q2)
    assert turnover_fraction(q1, q2) == 0.0
