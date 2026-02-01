#!/usr/bin/env python3
"""Rerun negative controls with fixed code"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

base_dir = Path(__file__).parent.parent
sys.path.insert(0, str(base_dir / "src"))

from run_all import run_experiment

def rerun_negative_controls():
    """Rerun all negative control runs"""
    plan_path = base_dir / "outputs" / "reports" / "experiment_plan_fast.json"
    
    if not plan_path.exists():
        print(f"ERROR: Plan file not found: {plan_path}")
        return
    
    with open(plan_path, 'r', encoding='utf-8') as f:
        plan = json.load(f)
    
    # Find negative control runs
    negative_control_runs = []
    for entry in plan:
        run_id = entry.get("run_id", "")
        if "negative_control" in run_id.lower():
            negative_control_runs.append(entry)
    
    print(f"Found {len(negative_control_runs)} negative control runs to rerun")
    print("Rerunning with fixed code...\n")
    
    results = {
        "success": [],
        "failed": []
    }
    
    for i, entry in enumerate(negative_control_runs, 1):
        run_id = entry.get("run_id")
        print(f"[{i}/{len(negative_control_runs)}] Running: {run_id}")
        
        try:
            result = run_experiment(entry, base_dir=str(base_dir / "outputs" / "runs"))
            
            if result.get("status") == "ok":
                results["success"].append(run_id)
                print(f"  SUCCESS")
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
    results_path = base_dir / "paper" / "rerun_negative_controls_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {results_path}")
    
    return results

if __name__ == "__main__":
    rerun_negative_controls()
