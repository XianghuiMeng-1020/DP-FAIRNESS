"""
STEP 5: Full rerun with REAL data - FRESH START
强制从头开始，跳过已完成的运行（会自动使用真实数据）
在终端运行此脚本查看实时进度
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
    
    if status_path.exists() and metrics_path.exists():
        try:
            status = json.load(open(status_path, encoding='utf-8'))
            metrics = json.load(open(metrics_path, encoding='utf-8'))
            if status.get("status") == "ok":
                # Verify it's using real data (not synthetic)
                n_train = metrics.get("dp_n_train", 0)
                dataset = metrics.get("dataset", "")
                synthetic_sizes = {"OULAD": 4000, "UCI697": 320, "HarvardX_PersonCourse": 2400}
                if dataset in synthetic_sizes and n_train == synthetic_sizes[dataset]:
                    return "synthetic"  # Mark as synthetic, will rerun
                return "completed"
        except:
            pass
    return "pending"

def main():
    """Run full grid with progress reporting"""
    plan_path = "outputs/reports/experiment_plan_fast.json"
    
    print("="*80)
    print("STEP 5: Full Rerun with REAL Data Only - FRESH START")
    print("="*80)
    print(f"Plan: {plan_path}")
    
    # Load plan
    plan = load_plan(plan_path)
    total_runs = len(plan)
    
    print(f"\nTotal runs in plan: {total_runs}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print("\nStarting full rerun...")
    print("(Will skip runs that are already completed with REAL data)")
    print("(Will rerun runs that used SYNTHETIC data)")
    print()
    
    completed = 0
    skipped_real = 0
    rerun_synthetic = 0
    failed = []
    last_run_id = None
    last_progress_time = time.time()
    start_time = time.time()
    
    for i, entry in enumerate(plan):
        run_id = entry["run_id"]
        last_run_id = run_id
        
        # Check if already completed with real data
        status = check_run_status(run_id)
        if status == "completed":
            skipped_real += 1
            completed += 1
            continue
        elif status == "synthetic":
            rerun_synthetic += 1
            # Will rerun this one
        
        # Print progress every 5 seconds or every 5 runs
        current_time = time.time()
        if (current_time - last_progress_time >= 5) or (i % 5 == 0):
            elapsed = current_time - start_time
            remaining = total_runs - completed
            rate = completed / elapsed if elapsed > 0 else 0
            eta = remaining / rate if rate > 0 else 0
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                  f"Progress: {completed}/{total_runs} ({completed*100/total_runs:.1f}%) | "
                  f"Skipped (real): {skipped_real} | Rerun (synth): {rerun_synthetic} | "
                  f"Failed: {len(failed)} | "
                  f"Rate: {rate*60:.1f}/min | ETA: {eta/60:.1f}min | "
                  f"Current: {run_id}")
            last_progress_time = current_time
        
        # Run experiment
        try:
            result = run_experiment(entry, base_dir="outputs/runs")
            
            if result["status"] == "ok":
                completed += 1
                # Verify it's using real data
                metrics = result.get("metrics", {})
                n_train = metrics.get("dp_n_train", 0)
                dataset = entry["dataset"]
                
                # Check it's not synthetic size
                synthetic_sizes = {"OULAD": 4000, "UCI697": 320, "HarvardX_PersonCourse": 2400}
                if dataset in synthetic_sizes and n_train == synthetic_sizes[dataset]:
                    print(f"\nERROR: {run_id} still has synthetic n_train={n_train}!")
                    print(f"This should not happen - synthetic fallback is disabled!")
                    sys.exit(1)
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
    print(f"Skipped (already had real data): {skipped_real}")
    print(f"Rerun (was synthetic): {rerun_synthetic}")
    print(f"Failed: {len(failed)}")
    print(f"Total time: {elapsed_total/60:.1f} minutes")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if failed:
        print(f"\nFailed runs:")
        for f in failed[:10]:
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
