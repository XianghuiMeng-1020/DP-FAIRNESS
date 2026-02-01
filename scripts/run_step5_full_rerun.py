"""
STEP 5: Full rerun with real data only
- Run full grid with progress reporting
- Fail-fast on errors
- Regenerate all tables and audits after completion
"""
import sys
import time
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.run_all import load_plan, run_experiment

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
    
    print(f"Total runs: {total_runs}")
    print(f"Start time: {datetime.now()}")
    print("="*80)
    
    completed = 0
    failed = []
    last_run_id = None
    last_progress_time = time.time()
    
    for i, entry in enumerate(plan):
        run_id = entry["run_id"]
        last_run_id = run_id
        
        # Print progress every 30 seconds
        current_time = time.time()
        if current_time - last_progress_time >= 30:
            print(f"\n[Progress] {completed}/{total_runs} completed | Last: {last_run_id} | Failures: {len(failed)}")
            last_progress_time = current_time
        
        # Run experiment
        try:
            result = run_experiment(entry, base_dir="outputs/runs")
            
            if result["status"] == "ok":
                completed += 1
            else:
                failed.append({
                    "run_id": run_id,
                    "error": result.get("error", "Unknown error"),
                    "config": entry
                })
                # Fail-fast: stop on first failure
                print(f"\n{'='*80}")
                print(f"FAILURE: Run {run_id} failed")
                print(f"Config: {json.dumps(entry, indent=2)}")
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
            print(f"Config: {json.dumps(entry, indent=2)}")
            print(f"Exception: {e}")
            import traceback
            traceback.print_exc()
            print(f"{'='*80}")
            sys.exit(1)
    
    # Final summary
    print(f"\n{'='*80}")
    print("STEP 5 COMPLETE")
    print(f"{'='*80}")
    print(f"Total runs: {total_runs}")
    print(f"Completed: {completed}")
    print(f"Failed: {len(failed)}")
    print(f"End time: {datetime.now()}")
    
    if failed:
        print(f"\nFailed runs:")
        for f in failed:
            print(f"  {f['run_id']}: {f['error']}")
        sys.exit(1)
    
    print("\nAll runs completed successfully!")
    return True

if __name__ == "__main__":
    main()
