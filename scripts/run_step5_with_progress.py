"""
STEP 5: Full rerun with real data only - WITH TERMINAL PROGRESS
Run this in terminal to see real-time progress
"""
import sys
import time
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.run_all import load_plan, run_experiment, get_run_dir

def check_run_status(run_id, base_dir="outputs/runs"):
    """Check if run is already completed"""
    run_dir = get_run_dir(run_id, base_dir)
    status_path = run_dir / "status.json"
    metrics_path = run_dir / "metrics.json"
    
    if status_path.exists():
        try:
            status = json.load(open(status_path, encoding='utf-8'))
            if status.get("status") == "ok" and metrics_path.exists():
                return "completed"
            elif status.get("status") == "failed":
                return "failed"
        except:
            pass
    return "pending"

def main():
    """Run full grid with progress reporting"""
    plan_path = "outputs/reports/experiment_plan_fast.json"
    
    print("="*80)
    print("STEP 5: Full Rerun with REAL Data Only")
    print("="*80)
    print(f"Plan: {plan_path}")
    
    # Load plan
    plan = load_plan(plan_path)
    total_runs = len(plan)
    
    # Check existing runs
    print("\nChecking existing runs...")
    existing_completed = 0
    existing_failed = 0
    for entry in plan[:100]:  # Check first 100
        status = check_run_status(entry["run_id"])
        if status == "completed":
            existing_completed += 1
        elif status == "failed":
            existing_failed += 1
    
    print(f"Total runs in plan: {total_runs}")
    print(f"Estimated existing completed: ~{existing_completed * total_runs // 100}")
    print(f"Estimated existing failed: ~{existing_failed * total_runs // 100}")
    print("="*80)
    
    print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Starting full rerun...\n")
    
    completed = 0
    skipped = 0
    failed = []
    last_run_id = None
    last_progress_time = time.time()
    start_time = time.time()
    
    for i, entry in enumerate(plan):
        run_id = entry["run_id"]
        last_run_id = run_id
        
        # Check if already completed
        status = check_run_status(run_id)
        if status == "completed":
            skipped += 1
            completed += 1
            continue
        
        # Print progress every 10 seconds or every 10 runs
        current_time = time.time()
        if (current_time - last_progress_time >= 10) or (i % 10 == 0):
            elapsed = current_time - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (total_runs - completed) / rate if rate > 0 else 0
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Progress: {completed}/{total_runs} completed ({completed*100/total_runs:.1f}%) | "
                  f"Skipped: {skipped} | Failed: {len(failed)} | "
                  f"Rate: {rate:.1f} runs/min | ETA: {eta/60:.1f} min | "
                  f"Last: {last_run_id}")
            last_progress_time = current_time
        
        # Run experiment
        try:
            result = run_experiment(entry, base_dir="outputs/runs")
            
            if result["status"] == "ok":
                completed += 1
                # Verify it's using real data
                metrics = result.get("metrics", {})
                n_train = metrics.get("dp_n_train", 0)
                if n_train > 0:
                    # Check it's not synthetic size
                    dataset = entry["dataset"]
                    synthetic_sizes = {"OULAD": 4000, "UCI697": 320, "HarvardX_PersonCourse": 2400}
                    if n_train == synthetic_sizes.get(dataset):
                        print(f"\nWARNING: {run_id} has suspicious n_train={n_train} (matches synthetic)")
            else:
                failed.append({
                    "run_id": run_id,
                    "error": result.get("error", "Unknown error"),
                    "config": entry
                })
                # Fail-fast: stop on first failure
                print(f"\n{'='*80}")
                print(f"FAILURE: Run {run_id} failed")
                print(f"Dataset: {entry.get('dataset')}")
                print(f"Error: {result.get('error', 'Unknown')}")
                print(f"{'='*80}")
                sys.exit(1)
                
        except Exception as e:
            failed.append({
                "run_id": run_id,
                "error": str(e),
                "config": entry
            })
            print(f"\n{'='*80}")
            print(f"EXCEPTION: Run {run_id} raised exception")
            print(f"Dataset: {entry.get('dataset')}")
            print(f"Exception: {e}")
            import traceback
            traceback.print_exc()
            print(f"{'='*80}")
            sys.exit(1)
    
    # Final summary
    elapsed_total = time.time() - start_time
    print(f"\n{'='*80}")
    print("STEP 5 COMPLETE")
    print(f"{'='*80}")
    print(f"Total runs: {total_runs}")
    print(f"Completed: {completed}")
    print(f"Skipped (already done): {skipped}")
    print(f"Failed: {len(failed)}")
    print(f"Total time: {elapsed_total/60:.1f} minutes")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if failed:
        print(f"\nFailed runs:")
        for f in failed[:10]:  # Show first 10
            print(f"  {f['run_id']}: {f['error'][:100]}")
        if len(failed) > 10:
            print(f"  ... and {len(failed)-10} more")
        sys.exit(1)
    
    print("\n✅ All runs completed successfully!")
    print("\nNext steps:")
    print("1. python src/reporting.py")
    print("2. python src/audit_fullpaper.py")
    print("3. python src/sanity_checks.py")
    return True

if __name__ == "__main__":
    main()
