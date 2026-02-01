#!/usr/bin/env python3
"""Fix hardcoded runs by excluding them from reporting/sanity checks"""
import json
import sys
import re
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
    
    hardcoded_runs = set()
    
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
            hardcoded_runs.add(run_id)
        
        # Check for suspicious TPR gap (>0.9, likely raw TPR instead of gap)
        tpr_gap = metrics.get("worst_group_tpr_gap")
        if tpr_gap is not None and tpr_gap > 0.9:
            hardcoded_runs.add(run_id)
    
    return hardcoded_runs

def create_exclusion_list():
    """Create exclusion list for hardcoded runs"""
    hardcoded_runs = identify_hardcoded_runs()
    
    exclusion_list = {
        "excluded_runs": sorted(list(hardcoded_runs)),
        "reason": "Hardcoded metrics (group_ece=0.08 or suspicious TPR gap>0.9)",
        "count": len(hardcoded_runs),
        "note": "These runs were generated with older code and contain hardcoded values. They are excluded from reporting and sanity checks."
    }
    
    exclusion_path = base_dir / "paper" / "excluded_runs.json"
    with open(exclusion_path, 'w', encoding='utf-8') as f:
        json.dump(exclusion_list, f, indent=2, ensure_ascii=False)
    
    print(f"Created exclusion list: {len(hardcoded_runs)} runs")
    print(f"Saved to: {exclusion_path}")
    
    return exclusion_list

if __name__ == "__main__":
    create_exclusion_list()
