"""
Smoke test: Run 3 test runs (one per dataset) to verify real data is used
"""
import sys
from pathlib import Path
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.run_all import load_plan, run_experiment

def smoke_test():
    """Run smoke test with one run per dataset"""
    print("="*80)
    print("SMOKE TEST: Verifying real data usage")
    print("="*80)
    
    # Create minimal test plan (one run per dataset)
    test_runs = [
        {"run_id": "smoke_test_oulad", "dataset": "OULAD", "model": "LR", "model_variant": None,
         "train_defense": "none", "publish_defense": None, "eps": None, "seed": 42},
        {"run_id": "smoke_test_uci697", "dataset": "UCI697", "model": "LR", "model_variant": None,
         "train_defense": "none", "publish_defense": None, "eps": None, "seed": 42},
        {"run_id": "smoke_test_harvardx", "dataset": "HarvardX_PersonCourse", "model": "LR", "model_variant": None,
         "train_defense": "none", "publish_defense": None, "eps": None, "seed": 42},
    ]
    
    results = {}
    
    for run_config in test_runs:
        dataset = run_config["dataset"]
        run_id = run_config["run_id"]
        
        print(f"\n{'='*80}")
        print(f"Testing {dataset} (run_id: {run_id})")
        print('='*80)
        
        try:
            result = run_experiment(run_config, base_dir="outputs/runs")
            
            if result["status"] == "ok":
                # Check artifacts
                run_dir = Path("outputs/runs") / run_id
                
                # Check n_total is not synthetic size
                metrics = result.get("metrics", {})
                n_train = metrics.get("dp_n_train", 0)
                n_test = metrics.get("n_test", 0)
                n_total = n_train + n_test
                
                print(f"\nDataset statistics:")
                print(f"  n_total: {n_total:,}")
                print(f"  n_train: {n_train:,}")
                print(f"  n_test: {n_test:,}")
                
                # Sanity check: not synthetic sizes
                synthetic_sizes = {"OULAD": 5000, "UCI697": 400, "HarvardX_PersonCourse": 3000}
                if n_total == synthetic_sizes.get(dataset):
                    print(f"\nERROR: n_total={n_total} matches synthetic size for {dataset}")
                    results[dataset] = False
                    continue
                
                # Check artifacts exist
                artifacts = {
                    "predictions_base.npy": run_dir / "predictions_base.npy",
                    "predictions_released.npy": run_dir / "predictions_released.npy",
                    "membership.npy": run_dir / "membership.npy",
                }
                
                print(f"\nChecking artifacts:")
                all_exist = True
                for name, path in artifacts.items():
                    exists = path.exists()
                    print(f"  {name}: {'EXISTS' if exists else 'MISSING'}")
                    if not exists:
                        all_exist = False
                
                if not all_exist:
                    print(f"\nERROR: Missing artifacts for {dataset}")
                    results[dataset] = False
                    continue
                
                # Check membership.npy size matches n_train + n_test
                membership = np.load(artifacts["membership.npy"])
                expected_size = n_train + n_test
                actual_size = len(membership)
                
                print(f"\nMembership size check:")
                print(f"  Expected: {expected_size:,} (n_train + n_test)")
                print(f"  Actual: {actual_size:,}")
                
                if actual_size != expected_size:
                    print(f"\nERROR: membership.npy size mismatch for {dataset}")
                    results[dataset] = False
                    continue
                
                print(f"\nSUCCESS: {dataset} smoke test passed")
                results[dataset] = True
                
            else:
                print(f"\nERROR: Run failed for {dataset}")
                print(f"Error: {result.get('error', 'Unknown error')}")
                results[dataset] = False
                
        except Exception as e:
            print(f"\nERROR: Exception during smoke test for {dataset}")
            import traceback
            traceback.print_exc()
            results[dataset] = False
    
    # Summary
    print("\n" + "="*80)
    print("SMOKE TEST SUMMARY")
    print("="*80)
    for dataset, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"{dataset:<30} {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\nSUCCESS: All smoke tests passed. Ready for full rerun!")
        return True
    else:
        print("\nERROR: Some smoke tests failed. DO NOT proceed with full rerun.")
        return False

if __name__ == "__main__":
    success = smoke_test()
    sys.exit(0 if success else 1)
