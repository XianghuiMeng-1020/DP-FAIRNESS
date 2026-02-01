"""
Run all remaining experiments with real-time terminal output
Simple script that runs in foreground so you can see progress
"""
import sys
import time
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.run_all import load_plan, run_experiment, get_run_dir

def is_completed(run_id):
    """Check if run is completed"""
    run_dir = get_run_dir(run_id)
    status_file = run_dir / "status.json"
    metrics_file = run_dir / "metrics.json"
    return status_file.exists() and metrics_file.exists() and json.load(open(status_file)).get("status") == "ok"

def main():
    plan = load_plan("outputs/reports/experiment_plan_fast.json")
    total = len(plan)
    
    # Count completed
    completed = sum(1 for e in plan if is_completed(e["run_id"]))
    remaining = total - completed
    
    print("="*70)
    print(f"Running Remaining Experiments")
    print(f"Total: {total} | Completed: {completed} | Remaining: {remaining}")
    print("="*70)
    print()
    
    start_time = time.time()
    last_progress = time.time()
    current_completed = completed
    
    for i, entry in enumerate(plan):
        run_id = entry["run_id"]
        
        if is_completed(run_id):
            continue
        
        # Print progress every 10 seconds
        now = time.time()
        if now - last_progress >= 10:
            elapsed = now - start_time
            rate = current_completed / elapsed if elapsed > 0 else 0
            eta = (remaining - (current_completed - completed)) / rate if rate > 0 else 0
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {current_completed}/{total} ({current_completed*100/total:.1f}%) | "
                  f"Rate: {rate*60:.1f}/min | ETA: {eta/60:.1f}min | Running: {run_id}")
            last_progress = now
        
        print(f"[{i+1}/{total}] {run_id} ({entry.get('dataset')})...", end=" ", flush=True)
        
        try:
            result = run_experiment(entry)
            if result["status"] == "ok":
                current_completed += 1
                print("OK")
            else:
                print(f"FAILED: {result.get('error', 'Unknown')[:80]}")
                return False
        except Exception as e:
            print(f"ERROR: {str(e)[:80]}")
            import traceback
            traceback.print_exc()
            return False
    
    elapsed = time.time() - start_time
    print()
    print("="*70)
    print(f"Complete! {current_completed}/{total} runs completed in {elapsed/60:.1f} minutes")
    print("="*70)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
