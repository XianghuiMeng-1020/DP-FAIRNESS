"""
Resume experiment run, skipping failed runs
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
    """Resume experiment, skipping failed runs"""
    plan_path = "outputs/reports/experiment_plan_fast.json"
    
    print("="*80)
    print("Resume Experiment Run (Skip Failed)")
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
    last_progress_time = time.time()
    
    for i, entry in enumerate(plan):
        run_id = entry["run_id"]
        status = check_run_status(run_id)
        
        if status == "completed":
            continue
        elif status == "failed":
            skipped_failed += 1
            print(f"[{i+1}/{total_runs}] Skipping {run_id} (previously failed)")
            continue
        
        # Print progress every 5 seconds
        current_time = time.time()
        if current_time - last_progress_time >= 5:
            elapsed = current_time - start_time
            rate = completed_count / elapsed if elapsed > 0 else 0
            remaining = total_runs - completed_count
            eta = remaining / rate if rate > 0 else 0
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                  f"Progress: {completed_count}/{total_runs} ({completed_count*100/total_runs:.1f}%) | "
                  f"Skipped failed: {skipped_failed} | "
                  f"Rate: {rate*60:.1f}/min | ETA: {eta/60:.1f}min | "
                  f"Current: {run_id}")
            last_progress_time = current_time
        
        # Run experiment
        try:
            print(f"[{i+1}/{total_runs}] Running {run_id}...")
            result = run_experiment(entry, base_dir="outputs/runs")
            
            if result["status"] == "ok":
                completed_count += 1
            else:
                new_failed.append({
                    "run_id": run_id,
                    "error": result.get("error", "Unknown"),
                    "dataset": entry.get("dataset")
                })
                print(f"  [FAILED] {run_id}: {result.get('error', 'Unknown')[:100]}")
                # Continue instead of failing fast
                
        except Exception as e:
            new_failed.append({
                "run_id": run_id,
                "error": str(e),
                "dataset": entry.get("dataset")
            })
            print(f"  [EXCEPTION] {run_id}: {str(e)[:100]}")
            # Continue instead of failing fast
    
    # Summary
    elapsed_total = time.time() - start_time
    print(f"\n{'='*80}")
    print("Resume Complete")
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
    
    return True

if __name__ == "__main__":
    main()
