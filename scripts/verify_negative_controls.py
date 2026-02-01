#!/usr/bin/env python3
"""Verify negative control results after rerun"""
import json
import sys
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

base_dir = Path(__file__).parent.parent
runs_dir = base_dir / "outputs" / "runs"

def load_metrics(run_id):
    """Load metrics.json for a run"""
    metrics_path = runs_dir / run_id / "metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def verify_negative_controls():
    """Verify negative control Test AUC values"""
    plan_path = base_dir / "outputs" / "reports" / "experiment_plan_fast.json"
    
    with open(plan_path, 'r', encoding='utf-8') as f:
        plan = json.load(f)
    
    random_labels_runs = []
    random_groups_runs = []
    
    for entry in plan:
        run_id = entry.get("run_id", "")
        if "random_labels" in run_id.lower():
            random_labels_runs.append(run_id)
        elif "random_groups" in run_id.lower():
            random_groups_runs.append(run_id)
    
    print("=== Verifying Negative Controls ===\n")
    
    # Check Random Labels
    print("Random Labels (should have Test AUC ~ 0.5):")
    random_labels_aucs = []
    for run_id in random_labels_runs:
        metrics = load_metrics(run_id)
        if metrics:
            test_auc = metrics.get("test_auc")
            if test_auc is not None:
                random_labels_aucs.append(test_auc)
                if abs(test_auc - 0.5) > 0.1:
                    print(f"  WARNING: {run_id}: Test AUC = {test_auc:.5f} (not near 0.5)")
    
    if random_labels_aucs:
        mean_auc = sum(random_labels_aucs) / len(random_labels_aucs)
        min_auc = min(random_labels_aucs)
        max_auc = max(random_labels_aucs)
        print(f"  Total runs: {len(random_labels_aucs)}")
        print(f"  Mean Test AUC: {mean_auc:.5f}")
        print(f"  Range: [{min_auc:.5f}, {max_auc:.5f}]")
        if abs(mean_auc - 0.5) <= 0.1:
            print(f"  ✓ PASS: Mean AUC is within tolerance of 0.5")
        else:
            print(f"  ✗ FAIL: Mean AUC {mean_auc:.5f} is not within tolerance of 0.5")
    
    # Check Random Groups
    print("\nRandom Groups (should have TPR Gap ~ 0):")
    random_groups_gaps = []
    for run_id in random_groups_runs:
        metrics = load_metrics(run_id)
        if metrics:
            tpr_gap = metrics.get("worst_group_tpr_gap")
            if tpr_gap is not None:
                random_groups_gaps.append(tpr_gap)
                if abs(tpr_gap) > 0.05:
                    print(f"  WARNING: {run_id}: TPR Gap = {tpr_gap:.5f} (not near 0)")
    
    if random_groups_gaps:
        mean_gap = sum(random_groups_gaps) / len(random_groups_gaps)
        min_gap = min(random_groups_gaps)
        max_gap = max(random_groups_gaps)
        print(f"  Total runs: {len(random_groups_gaps)}")
        print(f"  Mean TPR Gap: {mean_gap:.5f}")
        print(f"  Range: [{min_gap:.5f}, {max_gap:.5f}]")
        if abs(mean_gap) <= 0.05:
            print(f"  ✓ PASS: Mean gap is within tolerance of 0")
        else:
            print(f"  ✗ FAIL: Mean gap {mean_gap:.5f} is not within tolerance of 0")
    
    return {
        "random_labels": {
            "count": len(random_labels_aucs),
            "mean_auc": mean_auc if random_labels_aucs else None,
            "pass": abs(mean_auc - 0.5) <= 0.1 if random_labels_aucs else False
        },
        "random_groups": {
            "count": len(random_groups_gaps),
            "mean_gap": mean_gap if random_groups_gaps else None,
            "pass": abs(mean_gap) <= 0.05 if random_groups_gaps else False
        }
    }

if __name__ == "__main__":
    verify_negative_controls()
