"""
Rerun the 4 runs with seed consistency issues
"""
import sys
import io
import json
import shutil
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.run_all import load_plan, run_experiment, get_run_dir

# The problematic runs: OULAD|MLP|DP-SGD|output_coarsening|eps=5
# Seeds 1-5: fast_0211, fast_0212, fast_0213, fast_0214, fast_0215
problematic_run_ids = ["fast_0211", "fast_0212", "fast_0213", "fast_0214", "fast_0215"]

def main():
    plan = load_plan("outputs/reports/experiment_plan_fast.json")
    
    # Find entries for problematic runs
    runs_to_rerun = [e for e in plan if e["run_id"] in problematic_run_ids]
    
    print("="*70)
    print("Rerunning Seed Consistency Issue Runs")
    print("="*70)
    print(f"Found {len(runs_to_rerun)} runs to rerun")
    
    # Delete old run directories to force true rerun
    print("\nDeleting old run directories...")
    for entry in runs_to_rerun:
        run_id = entry["run_id"]
        run_dir = get_run_dir(run_id)
        if run_dir.exists():
            try:
                shutil.rmtree(run_dir)
                print(f"  Deleted {run_id}")
            except Exception as e:
                print(f"  ERROR deleting {run_id}: {e}")
    
    print("\nRerunning with fixed seed logic...")
    print("="*70)
    
    for i, entry in enumerate(runs_to_rerun):
        run_id = entry["run_id"]
        seed = entry.get("seed", 1)
        dataset = entry.get("dataset")
        
        print(f"\n[{i+1}/{len(runs_to_rerun)}] Running {run_id} (seed={seed}, {dataset})...")
        
        try:
            result = run_experiment(entry)
            
            if result["status"] == "ok":
                metrics = result.get("metrics", {})
                n_train = metrics.get("dp_n_train", 0)
                n_test = metrics.get("n_test", 0)
                test_auc = metrics.get("test_auc", 0)
                
                # Check if predictions are constant after label-only
                run_dir = get_run_dir(run_id)
                preds_released = None
                if (run_dir / "predictions_released.npy").exists():
                    import numpy as np
                    preds_released = np.load(run_dir / "predictions_released.npy")
                    unique_preds = len(np.unique(preds_released[:, 1]))
                    is_constant = unique_preds == 1
                else:
                    is_constant = None
                
                print(f"  OK: n_train={n_train}, n_test={n_test}, test_auc={test_auc:.4f}, "
                      f"predictions_constant={is_constant}")
            else:
                print(f"  FAILED: {result.get('error', 'Unknown')}")
                return False
                
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n" + "="*70)
    print("Rerun Complete!")
    print("="*70)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
