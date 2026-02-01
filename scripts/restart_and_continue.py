"""
Restart and continue remaining experiments
Ensures no duplicate runs, continues from where it stopped
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

def is_completed(run_id):
    """Check if run is completed"""
    run_dir = get_run_dir(run_id)
    status_file = run_dir / "status.json"
    metrics_file = run_dir / "metrics.json"
    
    if not (status_file.exists() and metrics_file.exists()):
        return False
    
    try:
        status = json.load(open(status_file, encoding='utf-8'))
        return status.get("status") == "ok"
    except:
        return False

def main():
    plan = load_plan("outputs/reports/experiment_plan_fast.json")
    total = len(plan)
    
    # Count completed
    completed = sum(1 for e in plan if is_completed(e["run_id"]))
    remaining = total - completed
    
    print("="*70)
    print("Restart and Continue Remaining Experiments")
    print(f"Total: {total} | Completed: {completed} | Remaining: {remaining}")
    print("="*70)
    print()
    
    if remaining == 0:
        print("All runs already completed!")
        return True
    
    start_time = time.time()
    last_progress = time.time()
    current_completed = completed
    failed_count = 0
    
    for i, entry in enumerate(plan):
        run_id = entry["run_id"]
        
        # Skip if already completed
        if is_completed(run_id):
            continue
        
        # Print progress every 10 seconds
        now = time.time()
        if now - last_progress >= 10:
            elapsed = now - start_time
            done_in_session = current_completed - completed
            if done_in_session > 0:
                rate = done_in_session / elapsed if elapsed > 0 else 0
                remaining_in_session = remaining - done_in_session
                eta = remaining_in_session / rate if rate > 0 else 0
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {current_completed}/{total} ({current_completed*100/total:.1f}%) | "
                      f"Session: {done_in_session} runs | Rate: {rate*60:.1f}/min | ETA: {eta/60:.1f}min | "
                      f"Running: {run_id}")
            last_progress = now
        
        print(f"[{i+1}/{total}] {run_id} ({entry.get('dataset')})...", end=" ", flush=True)
        
        try:
            result = run_experiment(entry)
            if result["status"] == "ok":
                current_completed += 1
                print("OK")
            else:
                failed_count += 1
                error_msg = result.get('error', 'Unknown')[:80]
                print(f"FAILED: {error_msg}")
                # Continue instead of stopping
        except Exception as e:
            failed_count += 1
            error_msg = str(e)[:80]
            print(f"ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            # Continue instead of stopping
    
    elapsed = time.time() - start_time
    print()
    print("="*70)
    print(f"Complete! {current_completed}/{total} runs completed")
    print(f"Session: {current_completed - completed} new runs in {elapsed/60:.1f} minutes")
    if failed_count > 0:
        print(f"Failed: {failed_count}")
    print("="*70)
    
    return failed_count == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
