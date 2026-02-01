#!/usr/bin/env python3
"""Identify runs with hardcoded metrics"""
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

def identify_hardcoded_runs():
    """Identify runs with hardcoded metrics"""
    plan_path = base_dir / "outputs" / "reports" / "experiment_plan_fast.json"
    
    with open(plan_path, 'r', encoding='utf-8') as f:
        plan = json.load(f)
    
    hardcoded_group_ece = []
    suspicious_tpr_gap = []
    
    for entry in plan:
        if not entry.get("is_core", False):
            continue
        
        run_id = entry.get("run_id")
        metrics = load_metrics(run_id)
        
        if not metrics:
            continue
        
        # Check for hardcoded group_ece = 0.08
        group_ece = metrics.get("group_ece")
        if group_ece is not None and abs(group_ece - 0.08) < 1e-6:
            hardcoded_group_ece.append(run_id)
        
        # Check for suspicious TPR gap (>0.9, likely raw TPR instead of gap)
        tpr_gap = metrics.get("worst_group_tpr_gap")
        if tpr_gap is not None and tpr_gap > 0.9:
            suspicious_tpr_gap.append({
                "run_id": run_id,
                "tpr_gap": tpr_gap,
                "dataset": entry.get("dataset"),
                "model": entry.get("model"),
                "train_defense": entry.get("train_defense"),
            })
    
    print("=== Identifying Hardcoded Runs ===\n")
    
    print(f"Runs with hardcoded group_ece=0.08: {len(hardcoded_group_ece)}")
    if len(hardcoded_group_ece) > 0:
        print("  Examples:")
        for run_id in hardcoded_group_ece[:10]:
            print(f"    - {run_id}")
        if len(hardcoded_group_ece) > 10:
            print(f"    ... and {len(hardcoded_group_ece) - 10} more")
    
    print(f"\nRuns with suspicious TPR gap (>0.9): {len(suspicious_tpr_gap)}")
    if len(suspicious_tpr_gap) > 0:
        print("  Examples:")
        for item in suspicious_tpr_gap[:10]:
            print(f"    - {item['run_id']}: TPR gap={item['tpr_gap']:.5f} ({item['dataset']}/{item['model']}/{item['train_defense']})")
        if len(suspicious_tpr_gap) > 10:
            print(f"    ... and {len(suspicious_tpr_gap) - 10} more")
    
    # Group by dataset/model/defense to see patterns
    print("\n=== Patterns ===")
    tpr_gap_by_setting = defaultdict(list)
    for item in suspicious_tpr_gap:
        key = (item['dataset'], item['model'], item['train_defense'])
        tpr_gap_by_setting[key].append(item['run_id'])
    
    print("Settings with suspicious TPR gap:")
    for (dataset, model, train_def), run_ids in sorted(tpr_gap_by_setting.items()):
        print(f"  {dataset}/{model}/{train_def}: {len(run_ids)} runs")
    
    return {
        "hardcoded_group_ece": hardcoded_group_ece,
        "suspicious_tpr_gap": suspicious_tpr_gap
    }

if __name__ == "__main__":
    identify_hardcoded_runs()
