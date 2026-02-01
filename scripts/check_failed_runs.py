"""
Check for failed or stuck runs
"""
import sys
import io
from pathlib import Path
import json
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def check_failed_runs():
    """Find failed runs"""
    plan_path = Path("outputs/reports/experiment_plan_fast.json")
    runs_dir = Path("outputs/runs")
    
    plan = json.load(open(plan_path, encoding='utf-8'))
    
    failed_runs = []
    stuck_runs = []
    
    for entry in plan:
        run_id = entry["run_id"]
        run_dir = runs_dir / run_id.replace("N/A", "N")
        status_file = run_dir / "status.json"
        
        if status_file.exists():
            try:
                status = json.load(open(status_file, encoding='utf-8'))
                if status.get("status") == "failed":
                    failed_runs.append({
                        "run_id": run_id,
                        "error": status.get("error", "Unknown"),
                        "dataset": entry.get("dataset")
                    })
            except:
                pass
        else:
            # Check if run directory exists but no status (might be stuck)
            if run_dir.exists():
                # Check if there are partial files
                has_partial = any(run_dir.glob("*.npy")) or (run_dir / "config.json").exists()
                if has_partial:
                    stuck_runs.append(run_id)
    
    print("="*80)
    print("Failed and Stuck Runs Check")
    print("="*80)
    
    if failed_runs:
        print(f"\nFailed runs: {len(failed_runs)}")
        for f in failed_runs[:10]:
            print(f"\n  Run ID: {f['run_id']}")
            print(f"  Dataset: {f['dataset']}")
            print(f"  Error: {f['error'][:200]}")
        if len(failed_runs) > 10:
            print(f"\n  ... and {len(failed_runs) - 10} more failed runs")
    else:
        print("\nNo failed runs found")
    
    if stuck_runs:
        print(f"\nPotentially stuck runs (have partial files but no status): {len(stuck_runs)}")
        print(f"First 10: {stuck_runs[:10]}")
    
    return failed_runs, stuck_runs

if __name__ == "__main__":
    failed, stuck = check_failed_runs()
    sys.exit(1 if (failed or stuck) else 0)
