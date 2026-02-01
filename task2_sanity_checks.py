"""
TASK 2: Sanity checks on 6 representative runs
"""
import json
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

def load_run_data(run_id, base_dir="outputs/runs"):
    """Load run artifacts"""
    run_dir = Path(base_dir) / run_id
    
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.json"
    predictions_path = run_dir / "predictions.npy"
    labels_path = run_dir / "test_labels.npy"
    groups_path = run_dir / "groups.npy"
    
    data = {}
    
    if config_path.exists():
        with open(config_path, "r") as f:
            data["config"] = json.load(f)
    
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            data["metrics"] = json.load(f)
    
    if predictions_path.exists():
        data["predictions"] = np.load(predictions_path)
    
    if labels_path.exists():
        data["labels"] = np.load(labels_path)
    
    if groups_path.exists():
        data["groups"] = np.load(groups_path)
    
    return data

def print_sanity_check(run_id, base_dir="outputs/runs"):
    """Print sanity check evidence for a run"""
    print(f"\n{'='*80}")
    print(f"RUN: {run_id}")
    print(f"{'='*80}")
    
    data = load_run_data(run_id, base_dir)
    
    if "config" not in data:
        print(f"ERROR: Config not found for {run_id}")
        return
    
    config = data["config"]
    print(f"Config: dataset={config.get('dataset')}, model={config.get('model')}, "
          f"train_def={config.get('train_defense')}, publish_def={config.get('publish_defense')}, "
          f"noise_type={config.get('noise_type')}, noise_scale={config.get('noise_scale')}")
    
    if "predictions" not in data or "labels" not in data:
        print(f"ERROR: Missing predictions or labels for {run_id}")
        return
    
    predictions = data["predictions"]
    labels = data["labels"]
    
    # Extract y_scores (positive class probability)
    if len(predictions.shape) > 1 and predictions.shape[1] == 2:
        y_scores_released = predictions[:, 1]
    else:
        y_scores_released = predictions.flatten()
    
    # Check if base predictions exist
    run_dir = Path(base_dir) / run_id
    base_predictions_path = run_dir / "predictions_base.npy"
    
    if base_predictions_path.exists():
        predictions_base = np.load(base_predictions_path)
        if len(predictions_base.shape) > 1 and predictions_base.shape[1] == 2:
            y_scores_base = predictions_base[:, 1]
        else:
            y_scores_base = predictions_base.flatten()
    else:
        # No base predictions saved - try to infer from released predictions
        # For runs without release defense, base == released
        if config.get("publish_defense") is None:
            print("\nNOTE: No predictions_base.npy found, but no release defense.")
            print("   Using released predictions as base (they should be identical).")
            y_scores_base = y_scores_released.copy()
        else:
            print("\nWARNING: predictions_base.npy not found!")
            print("   Cannot compare base vs released without base predictions.")
            print("   This run needs to be rerun with the fixed code.")
            y_scores_base = None
    
    print(f"\n1) y_scores summary stats:")
    if y_scores_base is not None:
        print(f"   BASE:")
        print(f"     mean={np.mean(y_scores_base):.6f}, std={np.std(y_scores_base):.6f}")
        print(f"     min={np.min(y_scores_base):.6f}, max={np.max(y_scores_base):.6f}")
        print(f"     unique_count={len(np.unique(y_scores_base))}")
    
    print(f"   RELEASED:")
    print(f"     mean={np.mean(y_scores_released):.6f}, std={np.std(y_scores_released):.6f}")
    print(f"     min={np.min(y_scores_released):.6f}, max={np.max(y_scores_released):.6f}")
    print(f"     unique_count={len(np.unique(y_scores_released))}")
    
    if y_scores_base is not None:
        print(f"\n2) Spearman rank correlation between base and released:")
        corr, pval = spearmanr(y_scores_base, y_scores_released)
        print(f"   correlation={corr:.6f}, p-value={pval:.6e}")
        if config.get("publish_defense") == "output_perturbation":
            if corr > 0.99:
                print(f"   WARNING: Correlation too high for perturbation! Should be < 0.99")
            else:
                print(f"   OK: Correlation reduced as expected for perturbation")
    
    print(f"\n3) Test AUC computed on:")
    if y_scores_base is not None:
        auc_base = roc_auc_score(labels, y_scores_base)
        print(f"   BASE: {auc_base:.6f}")
    auc_released = roc_auc_score(labels, y_scores_released)
    print(f"   RELEASED: {auc_released:.6f}")
    
    if "metrics" in data:
        metrics = data["metrics"]
        if "test_auc" in metrics:
            print(f"   METRICS.JSON: {metrics['test_auc']:.6f}")
            if abs(auc_released - metrics['test_auc']) > 1e-5:
                print(f"   WARNING: Mismatch between computed AUC and metrics.json!")
    
    if config.get("publish_defense") == "output_perturbation":
        if y_scores_base is not None:
            if auc_released > auc_base + 0.1:
                print(f"   CRITICAL: AUC increased dramatically with perturbation!")
                print(f"      Base AUC: {auc_base:.6f}, Released AUC: {auc_released:.6f}")
                print(f"      This suggests a bug!")
    
    if config.get("publish_defense") == "output_perturbation" and y_scores_base is not None:
        print(f"\n4) Perturbation verification:")
        noise = y_scores_released - y_scores_base
        print(f"   Noise distribution (released - base):")
        print(f"     mean={np.mean(noise):.6f} (should be ~0)")
        print(f"     std={np.std(noise):.6f} (should be ~0.1 for Gaussian)")
        print(f"     min={np.min(noise):.6f}, max={np.max(noise):.6f}")
        
        noise_type = config.get("noise_type", "gaussian")
        noise_scale = config.get("noise_scale", 0.1)
        
        if noise_type == "gaussian":
            expected_std = noise_scale
            if abs(np.std(noise) - expected_std) > 0.05:
                print(f"   WARNING: Noise std ({np.std(noise):.6f}) doesn't match expected ({expected_std:.6f})")
        elif noise_type == "laplace":
            # Laplace scale parameter is different from std
            expected_scale = noise_scale
            # For Laplace, std = scale * sqrt(2)
            expected_std = expected_scale * np.sqrt(2)
            if abs(np.std(noise) - expected_std) > 0.1:
                print(f"   WARNING: Noise std ({np.std(noise):.6f}) doesn't match expected Laplace scale")
        
        # Check clipping
        if np.any(y_scores_released < 0) or np.any(y_scores_released > 1):
            print(f"   WARNING: Released scores not clipped to [0,1]!")
        else:
            print(f"   OK: Scores properly clipped to [0,1]")
    
    if "groups" in data:
        print(f"\n5) Fairness sanity check:")
        groups = data["groups"]
        unique_groups = np.unique(groups)
        print(f"   Group sizes:")
        for g in unique_groups:
            group_mask = (groups == g)
            group_size = np.sum(group_mask)
            pos_count = np.sum(labels[group_mask] == 1)
            print(f"     Group {g}: {group_size} total, {pos_count} positive")
            
            if group_size == 0:
                print(f"     WARNING: Group {g} has zero samples!")
            if pos_count == 0:
                print(f"     WARNING: Group {g} has zero positive samples!")
        
        # Compute TPR/FPR/FNR per group
        y_pred = (y_scores_released >= 0.5).astype(int)
        for g in unique_groups:
            mask = (groups == g)
            group_true = labels[mask]
            group_pred = y_pred[mask]
            
            tp = np.sum((group_true == 1) & (group_pred == 1))
            fp = np.sum((group_true == 0) & (group_pred == 1))
            fn = np.sum((group_true == 1) & (group_pred == 0))
            tn = np.sum((group_true == 0) & (group_pred == 0))
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
            
            print(f"   Group {g}: TPR={tpr:.4f}, FPR={fpr:.4f}, FNR={fnr:.4f}")
            print(f"              Confusion matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        
        if "metrics" in data:
            metrics = data["metrics"]
            if "worst_group_tpr_gap" in metrics and metrics["worst_group_tpr_gap"] is not None:
                gap = metrics["worst_group_tpr_gap"]
                print(f"   Worst group TPR gap: {gap:.6f}")
                if gap > 0.3:
                    print(f"   WARNING: Extreme TPR gap > 0.3!")

def main():
    # 6 representative runs:
    # 1. OULAD LR, none
    # 2. OULAD LR, coarsening
    # 3. OULAD LR, perturbation
    # 4. OULAD MLP-small, none
    # 5. OULAD MLP-small, coarsening
    # 6. OULAD MLP-small, perturbation
    
    representative_runs = [
        "fast_0000",  # OULAD LR, none
        "fast_0012",  # OULAD LR, coarsening (need to find)
        "fast_0005",  # OULAD LR, perturbation
        "fast_0030",  # OULAD MLP-small, none
        "fast_0120",  # OULAD MLP-small, coarsening (need to find)
        "fast_0127",  # OULAD MLP-small, perturbation (need to find)
    ]
    
    # Find actual runs from plan
    plan_path = Path("outputs/reports/experiment_plan_fast.json")
    with open(plan_path, "r") as f:
        plan = json.load(f)
    
    # Find runs matching criteria
    found_runs = {}
    
    # OULAD LR, none
    for r in plan:
        if r["dataset"] == "OULAD" and r["model"] == "LR" and r.get("publish_defense") is None:
            found_runs["LR_none"] = r["run_id"]
            break
    
    # OULAD LR, coarsening
    for r in plan:
        if r["dataset"] == "OULAD" and r["model"] == "LR" and r.get("publish_defense") == "output_coarsening":
            found_runs["LR_coarsening"] = r["run_id"]
            break
    
    if "LR_coarsening" not in found_runs:
        print("WARNING: Could not find OULAD LR coarsening run")
    
    # OULAD LR, perturbation
    for r in plan:
        if r["dataset"] == "OULAD" and r["model"] == "LR" and r.get("publish_defense") == "output_perturbation":
            found_runs["LR_perturbation"] = r["run_id"]
            break
    
    # OULAD MLP-small, none
    for r in plan:
        if r["dataset"] == "OULAD" and r["model"] == "MLP" and r.get("model_variant") == "small" and r.get("publish_defense") is None:
            found_runs["MLP_small_none"] = r["run_id"]
            break
    
    # OULAD MLP-small, coarsening
    for r in plan:
        if r["dataset"] == "OULAD" and r["model"] == "MLP" and r.get("model_variant") == "small" and r.get("publish_defense") == "output_coarsening":
            found_runs["MLP_small_coarsening"] = r["run_id"]
            break
    
    # OULAD MLP-small, perturbation
    for r in plan:
        if r["dataset"] == "OULAD" and r["model"] == "MLP" and r.get("model_variant") == "small" and r.get("publish_defense") == "output_perturbation":
            found_runs["MLP_small_perturbation"] = r["run_id"]
            break
    
    print("Found representative runs:")
    for key, run_id in found_runs.items():
        print(f"  {key}: {run_id}")
    
    # Run sanity checks
    for key, run_id in found_runs.items():
        print_sanity_check(run_id)

if __name__ == "__main__":
    main()
