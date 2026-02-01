"""
Directly run remaining experiments - non-background version
Run this in terminal to see real-time output
"""
import sys
import io
import time
import json
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.run_all import load_plan, run_experiment, get_run_dir

def check_run_status(run_id, base_dir="outputs/runs"):
    """Check if run is completed"""
    run_dir = get_run_dir(run_id, base_dir)
    status_path = run_dir / "status.json"
    metrics_path = run_dir / "metrics.json"
    
    if status_path.exists() and metrics_path.exists():
        try:
            status = json.load(open(status_path, encoding='utf-8'))
            if status.get("status") == "ok":
                return "completed"
            elif status.get("status") == "failed":
                return "failed"
        except:
            pass
    return "pending"

def main():
    """Run remaining experiments"""
    plan_path = "outputs/reports/experiment_plan_fast.json"
    
    print("="*80)
    print("Running Remaining Experiments")
    print("="*80)
    
    plan = load_plan(plan_path)
    total_runs = len(plan)
    
    # Count status
    completed = sum(1 for e in plan if check_run_status(e["run_id"]) == "completed")
    failed = sum(1 for e in plan if check_run_status(e["run_id"]) == "failed")
    pending = total_runs - completed - failed
    
    print(f"Total runs: {total_runs}")
    print(f"Completed: {completed}")
    print(f"Failed: {failed} (will skip)")
    print(f"Pending: {pending}")
    print("="*80)
    print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    completed_count = completed
    skipped_failed = 0
    new_failed = []
    start_time = time.time()
    
    for i, entry in enumerate(plan):
        run_id = entry["run_id"]
        status = check_run_status(run_id)
        
        if status == "completed":
            continue
        elif status == "failed":
            skipped_failed += 1
            if skipped_failed <= 5 or skipped_failed % 20 == 0:
                print(f"[{i+1}/{total_runs}] Skipping {run_id} (previously failed)")
            continue
        
        # Run experiment
        print(f"[{i+1}/{total_runs}] Running {run_id} ({entry.get('dataset')})...")
        try:
            result = run_experiment(entry, base_dir="outputs/runs")
            
            if result["status"] == "ok":
                completed_count += 1
                print(f"  [OK] Completed ({completed_count}/{total_runs})")
            else:
                new_failed.append({
                    "run_id": run_id,
                    "error": result.get("error", "Unknown"),
                    "dataset": entry.get("dataset")
                })
                print(f"  [FAILED] {result.get('error', 'Unknown')[:100]}")
                
        except Exception as e:
            new_failed.append({
                "run_id": run_id,
                "error": str(e),
                "dataset": entry.get("dataset")
            })
            print(f"  [EXCEPTION] {str(e)[:100]}")
            import traceback
            traceback.print_exc()
    
    # Summary
    elapsed_total = time.time() - start_time
    print(f"\n{'='*80}")
    print("Complete")
    print(f"{'='*80}")
    print(f"Total runs: {total_runs}")
    print(f"Completed: {completed_count}")
    print(f"Skipped (previously failed): {skipped_failed}")
    print(f"New failures: {len(new_failed)}")
    print(f"Total time: {elapsed_total/60:.1f} minutes")
    
    if new_failed:
        print(f"\nNew failed runs:")
        for f in new_failed[:10]:
            print(f"  {f['run_id']} ({f['dataset']}): {f['error'][:100]}")

if __name__ == "__main__":
    main()
