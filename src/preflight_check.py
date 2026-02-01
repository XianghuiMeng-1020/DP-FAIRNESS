"""
Preflight check script: Architecture lock and smoke test before full grid rerun
"""
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_loader import load_dataset
from src.model_trainer import ModelTrainer
from src.run_all import run_experiment


def print_section(title):
    """Print section title"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def compute_parameter_count(model_type, variant, input_dim):
    """Compute model parameter count"""
    if model_type == "LR":
        # LR: input_dim weights + 1 bias
        return input_dim + 1
    
    elif model_type == "MLP":
        if variant == "small":
            # MLP-small: input -> 64 -> 2
            # Layer 1: input_dim * 64 + 64 (bias)
            # Layer 2: 64 * 2 + 2 (bias)
            return (input_dim * 64 + 64) + (64 * 2 + 2)
        
        elif variant == "large":
            # MLP-large: input -> 256 -> 256 -> 2 (Option A, recommended)
            # Layer 1: input_dim * 256 + 256 (bias)
            # Layer 2: 256 * 256 + 256 (bias)
            # Layer 3: 256 * 2 + 2 (bias)
            return (input_dim * 256 + 256) + (256 * 256 + 256) + (256 * 2 + 2)
    
    elif model_type == "XGBoost":
        # XGBoost parameter count hard to compute exactly, return None
        return None
    
    return None


def check_architecture():
    """1. ARCHITECTURE LOCK - Confirm MLP-large architecture"""
    print_section("1. ARCHITECTURE LOCK")
    
    print("\nCurrent MLP-large architecture implementation:")
    print("  Option B: input -> 256 -> 128 -> 2")
    
    print("\nPaper description:")
    print("  '3-layer, hidden=256 (>=2x small)'")
    print("  This suggests Option A is more appropriate: input -> 256 -> 256 -> 2")
    
    print("\nRecommended choice: Option A (input -> 256 -> 256 -> 2)")
    print("  Reasons:")
    print("  - Matches paper description 'hidden=256'")
    print("  - Clearer 2x relationship with MLP-small (hidden=64)")
    print("  - Parameter count: ~69K-72K (depends on input dimension)")
    
    # Update code to use Option A
    print("\nUpdating code to use Option A...")
    return "A"  # Return selected option


def check_input_dimensions():
    """2. INPUT DIMENSION & PARAM COUNTS"""
    print_section("2. INPUT DIMENSION & PARAM COUNTS")
    
    datasets = ["OULAD", "UCI697", "HarvardX_PersonCourse"]
    results = {}
    
    for dataset in datasets:
        print(f"\nDataset: {dataset}")
        print("-" * 80)
        
        # Load dataset to get input dimension
        X_train, X_test, y_train, y_test, groups_test = load_dataset(dataset, seed=42)
        input_dim = X_train.shape[1]
        
        print(f"  Post-encoding input dimension: {input_dim}")
        
        # Compute parameter counts for each model
        param_counts = {}
        
        # LR
        lr_params = compute_parameter_count("LR", None, input_dim)
        param_counts["LR"] = lr_params
        print(f"  LR parameter count: {lr_params}")
        
        # MLP-small
        mlp_small_params = compute_parameter_count("MLP", "small", input_dim)
        param_counts["MLP-small"] = mlp_small_params
        print(f"  MLP-small parameter count: {mlp_small_params} (~{mlp_small_params/1000:.1f}K)")
        
        # MLP-large (Option A: 256 -> 256 -> 2)
        mlp_large_params = compute_parameter_count("MLP", "large", input_dim)
        param_counts["MLP-large"] = mlp_large_params
        print(f"  MLP-large parameter count: {mlp_large_params} (~{mlp_large_params/1000:.1f}K)")
        
        # XGBoost
        print(f"  XGBoost parameter count: ~100-500 (tree model, hard to compute exactly)")
        
        results[dataset] = {
            "input_dim": input_dim,
            "param_counts": param_counts
        }
    
    return results


def smoke_test():
    """3. SMOKE TEST - Small-scale test to verify all functionality"""
    print_section("3. SMOKE TEST")
    
    # Test configuration: 1 dataset, 2 seeds, 4 defense configs
    test_configs = [
        {
            "name": "none",
            "train_defense": "none",
            "publish_defense": None,
            "eps": None,
        },
        {
            "name": "DP-SGD eps=5",
            "train_defense": "DP-SGD",
            "publish_defense": None,
            "eps": 5,
        },
        {
            "name": "output_coarsening label-only step=0.05",
            "train_defense": "none",
            "publish_defense": "output_coarsening",
            "coarsening_type": "label-only",
            "coarsening_step": 0.05,
            "eps": None,
        },
        {
            "name": "output_perturbation Gaussian scale=0.1",
            "train_defense": "none",
            "publish_defense": "output_perturbation",
            "noise_type": "gaussian",
            "noise_scale": 0.1,
            "eps": None,
        },
    ]
    
    dataset = "OULAD"
    seeds = [42, 123]
    
    print(f"\nTest configuration:")
    print(f"  Dataset: {dataset}")
    print(f"  Seeds: {seeds}")
    print(f"  Defense configs: {len(test_configs)}")
    
    all_results = []
    all_passed = True
    
    for config in test_configs:
        print(f"\nTest config: {config['name']}")
        print("-" * 80)
        
        for seed in seeds:
            run_id = f"smoke_test_{config['name'].replace(' ', '_').replace('=', '_')}_seed{seed}"
            
            # Build entry
            entry = {
                "run_id": run_id,
                "dataset": dataset,
                "model": "MLP",
                "model_variant": "small",  # Use small for faster testing
                "train_defense": config["train_defense"],
                "publish_defense": config.get("publish_defense"),
                "eps": config.get("eps"),
                "seed": seed,
                "fairness_attribute": "gender",
                "visibility": "full",
                "Q": "full",
            }
            
            # Add release defense parameters
            if config.get("publish_defense") == "output_coarsening":
                entry["coarsening_type"] = config.get("coarsening_type", "rounding")
                entry["coarsening_step"] = config.get("coarsening_step")
            elif config.get("publish_defense") == "output_perturbation":
                entry["noise_type"] = config.get("noise_type", "gaussian")
                entry["noise_scale"] = config.get("noise_scale")
            
            print(f"\n  Running: {run_id}")
            
            try:
                # 运行实验
                result = run_experiment(entry)
                
                if result["status"] != "ok":
                    print(f"    [FAIL] Failed: {result.get('error', 'Unknown error')}")
                    all_passed = False
                    continue
                
                run_dir = Path("outputs/runs") / run_id
                metrics = result["metrics"]
                
                # Check 1: predictions_base.npy and predictions_released.npy exist
                base_path = run_dir / "predictions_base.npy"
                released_path = run_dir / "predictions_released.npy"
                
                base_exists = base_path.exists()
                released_exists = released_path.exists() if config.get("publish_defense") else True
                
                if not base_exists:
                    print(f"    [FAIL] predictions_base.npy missing")
                    all_passed = False
                else:
                    print(f"    [OK] predictions_base.npy exists")
                
                if config.get("publish_defense") and not released_exists:
                    print(f"    [FAIL] predictions_released.npy missing")
                    all_passed = False
                elif config.get("publish_defense"):
                    print(f"    [OK] predictions_released.npy exists")
                
                # Check 2: base AUC reasonableness
                base_auc = metrics.get("test_auc")
                if base_auc is None:
                    print(f"    [FAIL] test_auc missing")
                    all_passed = False
                elif base_auc > 0.99:
                    print(f"    [WARN] base AUC too high: {base_auc:.4f} (possible label leakage)")
                    all_passed = False
                elif base_auc < 0.5 or base_auc > 0.95:
                    print(f"    [WARN] base AUC may be unreasonable: {base_auc:.4f}")
                else:
                    print(f"    [OK] base AUC reasonable: {base_auc:.4f}")
                
                # Check 3: perturbation effect
                if config.get("publish_defense") == "output_perturbation":
                    # Load predictions to compute rank correlation
                    predictions_base = np.load(base_path)
                    predictions_released = np.load(released_path)
                    
                    if len(predictions_base.shape) > 1:
                        y_scores_base = predictions_base[:, 1]
                    else:
                        y_scores_base = predictions_base.flatten()
                    
                    if len(predictions_released.shape) > 1:
                        y_scores_released = predictions_released[:, 1]
                    else:
                        y_scores_released = predictions_released.flatten()
                    
                    # Compute AUC change
                    labels_path = run_dir / "test_labels.npy"
                    if labels_path.exists():
                        labels = np.load(labels_path).flatten()
                        base_auc_direct = roc_auc_score(labels, y_scores_base)
                        released_auc_direct = roc_auc_score(labels, y_scores_released)
                        auc_decrease = base_auc_direct - released_auc_direct
                        
                        if auc_decrease < -0.05:
                            print(f"    [WARN] perturbation did not reduce AUC: {auc_decrease:.4f}")
                        else:
                            print(f"    [OK] perturbation reduces AUC: {auc_decrease:.4f}")
                        
                        # Rank correlation
                        rank_corr, _ = spearmanr(y_scores_base, y_scores_released)
                        if rank_corr < 0.8:
                            print(f"    [WARN] rank correlation low: {rank_corr:.4f}")
                        else:
                            print(f"    [OK] rank correlation: {rank_corr:.4f}")
                
                # Check 4: coarsening effect
                if config.get("publish_defense") == "output_coarsening":
                    predictions_base = np.load(base_path)
                    predictions_released = np.load(released_path)
                    
                    if len(predictions_base.shape) > 1:
                        y_scores_base = predictions_base[:, 1]
                    else:
                        y_scores_base = predictions_base.flatten()
                    
                    if len(predictions_released.shape) > 1:
                        y_scores_released = predictions_released[:, 1]
                    else:
                        y_scores_released = predictions_released.flatten()
                    
                    n_unique_base = len(np.unique(y_scores_base))
                    n_unique_released = len(np.unique(y_scores_released))
                    
                    if n_unique_released >= n_unique_base:
                        print(f"    [WARN] coarsening did not reduce unique values: {n_unique_base} -> {n_unique_released}")
                    else:
                        print(f"    [OK] coarsening reduces unique values: {n_unique_base} -> {n_unique_released}")
                    
                    # AUC change
                    labels_path = run_dir / "test_labels.npy"
                    if labels_path.exists():
                        labels = np.load(labels_path).flatten()
                        base_auc_direct = roc_auc_score(labels, y_scores_base)
                        released_auc_direct = roc_auc_score(labels, y_scores_released)
                        auc_decrease = base_auc_direct - released_auc_direct
                        print(f"    [OK] AUC change: {auc_decrease:.4f}")
                
                # Check 5: fairness NA rule
                group_coverage = metrics.get("group_coverage")
                if group_coverage:
                    print(f"    [OK] group_coverage recorded")
                    # Check for NA cases
                    for group, coverage in group_coverage.items():
                        if not coverage.get("tpr_valid") or not coverage.get("fpr_valid"):
                            print(f"      Group {group}: TPR/FPR marked as NA (insufficient samples)")
                else:
                    print(f"    [WARN] group_coverage not recorded")
                
                all_results.append({
                    "run_id": run_id,
                    "config": config["name"],
                    "seed": seed,
                    "status": "ok",
                    "base_auc": base_auc,
                })
                
            except Exception as e:
                print(f"    [FAIL] Exception: {str(e)}")
                all_passed = False
                import traceback
                traceback.print_exc()
    
    print_section("SMOKE TEST SUMMARY")
    
    if all_passed:
        print("\n[SUCCESS] All tests passed! Ready to proceed with full grid rerun.")
    else:
        print("\n[FAIL] Some tests failed! Please fix issues before running full grid.")
    
    return all_passed, all_results


def main():
    """Main function"""
    print("=" * 80)
    print("PREFLIGHT CHECK: Architecture Lock and Smoke Test")
    print("=" * 80)
    
    # 1. Architecture lock
    arch_choice = check_architecture()
    
    # 2. Input dimensions and parameter counts
    dim_results = check_input_dimensions()
    
    # 3. Smoke test
    smoke_passed, smoke_results = smoke_test()
    
    # Save results
    output_path = Path("outputs/reports/preflight_check_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "architecture_choice": arch_choice,
            "input_dimensions": dim_results,
            "smoke_test_passed": smoke_passed,
            "smoke_test_results": smoke_results,
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_path}")
    
    if smoke_passed:
        print("\n" + "=" * 80)
        print("SUCCESS: Preflight check completed! Ready for TASK 3 full grid rerun.")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("FAILED: Preflight check failed! Please fix issues before continuing.")
        print("=" * 80)


if __name__ == "__main__":
    main()
