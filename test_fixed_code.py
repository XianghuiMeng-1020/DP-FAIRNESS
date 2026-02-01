"""
Test the fixed code by rerunning one perturbation run
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path("src").absolute()))
from run_all import run_experiment

# Test run: OULAD LR with perturbation
test_entry = {
    "run_id": "test_perturbation_fix",
    "dataset": "OULAD",
    "model": "LR",
    "model_variant": None,
    "train_defense": "none",
    "publish_defense": "output_perturbation",
    "eps": None,
    "visibility": "full",
    "Q": 5,
    "seed": 1,
    "fairness_attribute": "gender",
    "coarsening_type": None,
    "coarsening_step": None,
    "noise_type": "gaussian",
    "noise_scale": 0.1,
    "intensity": "low",
    "is_core": True,
    "is_diagnostic": False
}

print("Running test perturbation run...")
result = run_experiment(test_entry, base_dir="outputs/runs")

if result["status"] == "ok":
    print("\nSUCCESS: Run completed")
    metrics = result["metrics"]
    print(f"Test AUC: {metrics['test_auc']:.6f}")
    
    # Check if base and released predictions exist
    run_dir = Path("outputs/runs") / "test_perturbation_fix"
    base_path = run_dir / "predictions_base.npy"
    released_path = run_dir / "predictions_released.npy"
    
    if base_path.exists() and released_path.exists():
        print("OK: Both predictions_base.npy and predictions_released.npy exist")
        
        import numpy as np
        from sklearn.metrics import roc_auc_score
        from scipy.stats import spearmanr
        
        base_preds = np.load(base_path)
        released_preds = np.load(released_path)
        labels = np.load(run_dir / "test_labels.npy")
        
        y_scores_base = base_preds[:, 1]
        y_scores_released = released_preds[:, 1]
        
        auc_base = roc_auc_score(labels, y_scores_base)
        auc_released = roc_auc_score(labels, y_scores_released)
        
        print(f"\nAUC comparison:")
        print(f"  Base: {auc_base:.6f}")
        print(f"  Released: {auc_released:.6f}")
        print(f"  Difference: {auc_released - auc_base:.6f}")
        
        if auc_released > auc_base + 0.05:
            print("  WARNING: AUC increased significantly with perturbation!")
        else:
            print("  OK: AUC change is reasonable")
        
        corr, _ = spearmanr(y_scores_base, y_scores_released)
        print(f"\nSpearman correlation: {corr:.6f}")
        
        noise = y_scores_released - y_scores_base
        print(f"\nNoise statistics:")
        print(f"  Mean: {np.mean(noise):.6f} (should be ~0)")
        print(f"  Std: {np.std(noise):.6f} (should be ~0.1)")
        print(f"  Min: {np.min(noise):.6f}, Max: {np.max(noise):.6f}")
    else:
        print("ERROR: Base or released predictions not found!")
else:
    print(f"\nFAILED: {result.get('error', 'Unknown error')}")
