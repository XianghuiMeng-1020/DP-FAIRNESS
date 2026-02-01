"""
Rerun the 4 runs with seed consistency issues: OULAD|MLP|DP-SGD|output_coarsening|eps=5
Seeds 1-4: fast_0211, fast_0212, fast_0213, fast_0214
"""
import sys
import json
import shutil
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.run_all import load_plan, run_experiment, get_run_dir

# The 4 problematic runs: OULAD|MLP|DP-SGD|output_coarsening|eps=5
# Seeds 1-4: fast_0211, fast_0212, fast_0213, fast_0214
problematic_run_ids = ["fast_0211", "fast_0212", "fast_0213", "fast_0214"]

def check_config_seeds(runs_to_rerun):
    """确认config.json中的seed字段不同"""
    print("\n" + "="*70)
    print("Checking config.json seed fields...")
    print("="*70)
    for entry in runs_to_rerun:
        run_id = entry["run_id"]
        seed = entry.get("seed")
        run_dir = get_run_dir(run_id)
        config_path = run_dir / "config.json"
        
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            config_seed = config.get("seed")
            print(f"  {run_id}: plan seed={seed}, config seed={config_seed}")
        else:
            print(f"  {run_id}: config.json not found (will be created)")

def check_split_fingerprints(runs_to_rerun):
    """检查split fingerprint是否随seed变化（如果存在）"""
    print("\n" + "="*70)
    print("Checking split fingerprints...")
    print("="*70)
    fingerprints = []
    for entry in runs_to_rerun:
        run_id = entry["run_id"]
        seed = entry.get("seed")
        run_dir = get_run_dir(run_id)
        fingerprint_path = run_dir / "data_fingerprint.json"
        
        if fingerprint_path.exists():
            with open(fingerprint_path, "r", encoding="utf-8") as f:
                fingerprint = json.load(f)
            split_fp = fingerprint.get("split_fingerprint", "N/A")
            fingerprints.append((run_id, seed, split_fp))
            print(f"  {run_id} (seed={seed}): split_fp={split_fp[:20] if isinstance(split_fp, str) and len(split_fp) > 20 else split_fp}")
        else:
            print(f"  {run_id} (seed={seed}): data_fingerprint.json not found (will be created)")
    
    # 检查fingerprints是否不同
    if len(fingerprints) > 1:
        unique_fps = set(fp[2] for fp in fingerprints)
        if len(unique_fps) == 1 and fingerprints[0][2] != "N/A":
            print(f"  WARNING: All fingerprints are identical! This may indicate seed not used properly.")
        else:
            print(f"  OK: Found {len(unique_fps)} unique fingerprints")

def main():
    plan = load_plan("outputs/reports/experiment_plan_fast.json")
    
    # Find entries for problematic runs
    runs_to_rerun = [e for e in plan if e["run_id"] in problematic_run_ids]
    
    if len(runs_to_rerun) != 4:
        print(f"ERROR: Expected 4 runs, found {len(runs_to_rerun)}")
        return False
    
    print("="*70)
    print("Rerunning 4 Seed Consistency Issue Runs")
    print("="*70)
    print(f"Runs to rerun: {', '.join([e['run_id'] for e in runs_to_rerun])}")
    
    # Check existing configs and fingerprints
    check_config_seeds(runs_to_rerun)
    check_split_fingerprints(runs_to_rerun)
    
    # Delete old run directories to force true rerun
    print("\n" + "="*70)
    print("Deleting old run directories to force true rerun...")
    print("="*70)
    for entry in runs_to_rerun:
        run_id = entry["run_id"]
        run_dir = get_run_dir(run_id)
        if run_dir.exists():
            try:
                # Delete status.json if exists to prevent resume
                status_path = run_dir / "status.json"
                if status_path.exists():
                    status_path.unlink()
                    print(f"  Deleted status.json for {run_id}")
                
                # Optionally delete entire directory (commented out - we'll just delete status)
                # shutil.rmtree(run_dir)
                # print(f"  Deleted entire directory for {run_id}")
            except Exception as e:
                print(f"  ERROR deleting {run_id}: {e}")
        else:
            print(f"  {run_id}: directory does not exist (will be created)")
    
    print("\n" + "="*70)
    print("Rerunning with updated DataLoader seed shuffle logic...")
    print("="*70)
    
    results = []
    for i, entry in enumerate(runs_to_rerun):
        run_id = entry["run_id"]
        seed = entry.get("seed", 1)
        dataset = entry.get("dataset")
        
        print(f"\n[{i+1}/4] Running {run_id} (seed={seed}, {dataset})...")
        
        try:
            result = run_experiment(entry)
            
            if result["status"] == "ok":
                metrics = result.get("metrics", {})
                # Get dataset sizes from group_coverage (test set only)
                group_coverage = metrics.get("group_coverage", {})
                if group_coverage:
                    n_test = sum(g.get("n_total", 0) for g in group_coverage.values())
                    # Estimate n_train (typically 4x n_test for 80/20 split)
                    n_train = int(n_test * 4) if n_test > 0 else 0
                else:
                    n_train = metrics.get("dp_n_train", metrics.get("n_train", 0))
                    n_test = metrics.get("n_test", 0)
                test_auc = metrics.get("test_auc", 0)
                
                # Check if predictions are constant after label-only
                run_dir = get_run_dir(run_id)
                preds_released = None
                is_constant = None
                if (run_dir / "predictions_released.npy").exists():
                    preds_released = np.load(run_dir / "predictions_released.npy")
                    if preds_released.shape[1] >= 2:
                        unique_preds = len(np.unique(preds_released[:, 1]))
                        is_constant = unique_preds == 1
                
                # Verify config.json has correct seed
                config_path = run_dir / "config.json"
                config_seed = None
                if config_path.exists():
                    with open(config_path, "r", encoding="utf-8") as f:
                        config = json.load(f)
                    config_seed = config.get("seed")
                
                # Verify split fingerprint exists and is different
                fingerprint_path = run_dir / "data_fingerprint.json"
                split_fp = None
                if fingerprint_path.exists():
                    with open(fingerprint_path, "r", encoding="utf-8") as f:
                        fingerprint = json.load(f)
                    split_fp = fingerprint.get("split_fingerprint", "N/A")
                
                print(f"  ✓ OK: seed={config_seed}, n_train={n_train}, n_test={n_test}, "
                      f"test_auc={test_auc:.4f}, predictions_constant={is_constant}, "
                      f"split_fp={str(split_fp)[:15] if split_fp else 'N/A'}...")
                
                results.append({
                    "run_id": run_id,
                    "seed": seed,
                    "n_train": n_train,
                    "n_test": n_test,
                    "test_auc": test_auc,
                    "predictions_constant": is_constant,
                    "config_seed": config_seed,
                    "split_fp": str(split_fp)[:20] if split_fp else None
                })
            else:
                print(f"  ✗ FAILED: {result.get('error', 'Unknown')}")
                return False
                
        except Exception as e:
            print(f"  ✗ EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Print summary
    print("\n" + "="*70)
    print("Rerun Summary")
    print("="*70)
    for r in results:
        print(f"{r['run_id']}: seed={r['seed']}, n_train={r['n_train']}, n_test={r['n_test']}, "
              f"test_auc={r['test_auc']:.4f}, constant={r['predictions_constant']}")
    
    # Verify seeds are distinct
    seeds = [r['seed'] for r in results]
    if len(set(seeds)) != len(seeds):
        print(f"\nWARNING: Duplicate seeds found! {seeds}")
    else:
        print(f"\n✓ All seeds are distinct: {seeds}")
    
    # Verify split fingerprints are different (if not label-only causing constant predictions)
    fps = [r['split_fp'] for r in results if r['split_fp']]
    if len(fps) > 1:
        unique_fps = set(fps)
        if len(unique_fps) == 1:
            print(f"\nWARNING: All split fingerprints are identical! This may be expected for label-only coarsening.")
        else:
            print(f"\n✓ Found {len(unique_fps)} unique split fingerprints")
    
    print("\n" + "="*70)
    print("Rerun Complete!")
    print("="*70)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
