#!/usr/bin/env python3
"""Rerun runs with hardcoded metrics"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

base_dir = Path(__file__).parent.parent
sys.path.insert(0, str(base_dir / "src"))

from run_all import run_experiment

def load_metrics(run_id):
    """Load metrics.json for a run"""
    metrics_path = base_dir / "outputs" / "runs" / run_id / "metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def rerun_hardcoded_runs():
    """Rerun runs with hardcoded metrics"""
    plan_path = base_dir / "outputs" / "reports" / "experiment_plan_fast.json"
    
    with open(plan_path, 'r', encoding='utf-8') as f:
        plan = json.load(f)
    
    # Find runs with hardcoded metrics
    runs_to_rerun = []
    
    for entry in plan:
        if not entry.get("is_core", False):
            continue
        
        run_id = entry.get("run_id")
        metrics = load_metrics(run_id)
        
        if not metrics:
            continue
        
        # Check for hardcoded group_ece = 0.08 or suspicious TPR gap
        group_ece = metrics.get("group_ece")
        tpr_gap = metrics.get("worst_group_tpr_gap")
        
        should_rerun = False
        reason = []
        
        if group_ece is not None and abs(group_ece - 0.08) < 1e-6:
            should_rerun = True
            reason.append("hardcoded_group_ece")
        
        if tpr_gap is not None and tpr_gap > 0.9:
            should_rerun = True
            reason.append("suspicious_tpr_gap")
        
        if should_rerun:
            runs_to_rerun.append({
                "entry": entry,
                "run_id": run_id,
                "reason": reason
            })
    
    print(f"Found {len(runs_to_rerun)} runs with hardcoded metrics to rerun")
    print("Rerunning with fixed code...\n")
    
    results = {
        "success": [],
        "failed": []
    }
    
    for i, item in enumerate(runs_to_rerun, 1):
        entry = item["entry"]
        run_id = item["run_id"]
        reason = item["reason"]
        
        print(f"[{i}/{len(runs_to_rerun)}] Running: {run_id} (reason: {', '.join(reason)})")
        
        try:
            result = run_experiment(entry, base_dir=str(base_dir / "outputs" / "runs"))
            
            if result.get("status") == "ok":
                results["success"].append(run_id)
                # Verify fix
                new_metrics = load_metrics(run_id)
                if new_metrics:
                    group_ece = new_metrics.get("group_ece")
                    tpr_gap = new_metrics.get("worst_group_tpr_gap")
                    if group_ece is not None and abs(group_ece - 0.08) < 1e-6:
                        print(f"  WARNING: Still has hardcoded group_ece!")
                    elif tpr_gap is not None and tpr_gap > 0.9:
                        print(f"  WARNING: Still has suspicious TPR gap!")
                    else:
                        print(f"  SUCCESS - Fixed")
            else:
                results["failed"].append({"run_id": run_id, "error": result.get("error", "Unknown error")})
                print(f"  FAILED: {result.get('error', 'Unknown error')}")
        except Exception as e:
            results["failed"].append({"run_id": run_id, "error": str(e)})
            print(f"  EXCEPTION: {e}")
    
    print(f"\n=== Rerun Complete ===")
    print(f"Success: {len(results['success'])}")
    print(f"Failed: {len(results['failed'])}")
    
    # Save results
    results_path = base_dir / "paper" / "rerun_hardcoded_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {results_path}")
    
    return results

if __name__ == "__main__":
    rerun_hardcoded_runs()
