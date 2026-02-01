"""
TASK B: Check representative runs to prove perturbation outputs are evaluated correctly
Compare base vs released AUC for perturbation runs
"""
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

def load_plan(plan_path="outputs/reports/experiment_plan_fast.json"):
    with open(plan_path, "r", encoding="utf-8") as f:
        return json.load(f)

def find_runs(plan, dataset, model, train_defense, publish_defense=None, noise_type=None, noise_scale=None):
    """Find runs matching criteria"""
    results = []
    for entry in plan:
        if entry["dataset"] != dataset:
            continue
        if entry["model"] != model:
            continue
        if entry.get("train_defense") != train_defense:
            continue
        if publish_defense is None:
            if entry.get("publish_defense") is not None:
                continue
        else:
            if entry.get("publish_defense") != publish_defense:
                continue
            if noise_type and entry.get("noise_type") != noise_type:
                continue
            if noise_scale is not None and entry.get("noise_scale") != noise_scale:
                continue
        results.append(entry)
    return results

def compute_auc_from_predictions(predictions_path, labels_path):
    """Compute AUC from predictions file"""
    predictions = np.load(predictions_path)
    labels = np.load(labels_path)
    
    # Extract positive class probability
    if len(predictions.shape) > 1:
        if predictions.shape[1] == 2:
            y_scores = predictions[:, 1]
        else:
            y_scores = predictions.flatten()
    else:
        y_scores = predictions.flatten()
    
    y_true = labels.flatten()
    
    # Ensure binary classification
    unique_labels = np.unique(y_true)
    if len(unique_labels) != 2:
        return None
    
    return roc_auc_score(y_true, y_scores)

def check_run_pair(base_run, pert_run, base_dir="outputs/runs"):
    """Check a pair of runs (none vs perturbation)"""
    base_dir_path = Path(base_dir)
    
    base_run_dir = base_dir_path / base_run["run_id"]
    pert_run_dir = base_dir_path / pert_run["run_id"]
    
    results = {
        "base_run_id": base_run["run_id"],
        "pert_run_id": pert_run["run_id"],
        "base_auc_base": None,
        "base_auc_released": None,
        "pert_auc_base": None,
        "pert_auc_released": None,
        "base_rank_corr": None,
        "pert_rank_corr": None,
    }
    
    # Check base run (none defense)
    base_base_pred = base_run_dir / "predictions_base.npy"
    base_released_pred = base_run_dir / "predictions_released.npy"
    base_labels = base_run_dir / "test_labels.npy"
    
    if base_base_pred.exists() and base_labels.exists():
        results["base_auc_base"] = compute_auc_from_predictions(base_base_pred, base_labels)
    
    if base_released_pred.exists() and base_labels.exists():
        results["base_auc_released"] = compute_auc_from_predictions(base_released_pred, base_labels)
    
    # Check perturbation run
    pert_base_pred = pert_run_dir / "predictions_base.npy"
    pert_released_pred = pert_run_dir / "predictions_released.npy"
    pert_labels = pert_run_dir / "test_labels.npy"
    
    if pert_base_pred.exists() and pert_labels.exists():
        results["pert_auc_base"] = compute_auc_from_predictions(pert_base_pred, pert_labels)
    
    if pert_released_pred.exists() and pert_labels.exists():
        results["pert_auc_released"] = compute_auc_from_predictions(pert_released_pred, pert_labels)
    
    # Compute rank correlations
    if base_base_pred.exists() and base_released_pred.exists() and base_labels.exists():
        base_base = np.load(base_base_pred)
        base_released = np.load(base_released_pred)
        if len(base_base.shape) > 1:
            base_base_scores = base_base[:, 1] if base_base.shape[1] == 2 else base_base.flatten()
        else:
            base_base_scores = base_base.flatten()
        if len(base_released.shape) > 1:
            base_released_scores = base_released[:, 1] if base_released.shape[1] == 2 else base_released.flatten()
        else:
            base_released_scores = base_released.flatten()
        if len(base_base_scores) == len(base_released_scores):
            corr, _ = spearmanr(base_base_scores, base_released_scores)
            results["base_rank_corr"] = corr
    
    if pert_base_pred.exists() and pert_released_pred.exists() and pert_labels.exists():
        pert_base = np.load(pert_base_pred)
        pert_released = np.load(pert_released_pred)
        if len(pert_base.shape) > 1:
            pert_base_scores = pert_base[:, 1] if pert_base.shape[1] == 2 else pert_base.flatten()
        else:
            pert_base_scores = pert_base.flatten()
        if len(pert_released.shape) > 1:
            pert_released_scores = pert_released[:, 1] if pert_released.shape[1] == 2 else pert_released.flatten()
        else:
            pert_released_scores = pert_released.flatten()
        if len(pert_base_scores) == len(pert_released_scores):
            corr, _ = spearmanr(pert_base_scores, pert_released_scores)
            results["pert_rank_corr"] = corr
    
    return results

def main():
    plan = load_plan()
    base_dir = "outputs/runs"
    
    print("=" * 80)
    print("TASK B: Representative Run Evidence Check")
    print("=" * 80)
    print()
    
    # Find representative runs
    print("Finding representative runs...")
    
    # OULAD LR: none vs perturbation
    oulad_lr_none = find_runs(plan, "OULAD", "LR", "none", publish_defense=None)
    oulad_lr_pert = find_runs(plan, "OULAD", "LR", "none", publish_defense="output_perturbation", 
                              noise_type="gaussian", noise_scale=0.1)
    
    # OULAD XGBoost: none vs perturbation
    oulad_xgb_none = find_runs(plan, "OULAD", "XGBoost", "none", publish_defense=None)
    oulad_xgb_pert = find_runs(plan, "OULAD", "XGBoost", "none", publish_defense="output_perturbation",
                               noise_type="gaussian", noise_scale=0.1)
    
    # OULAD MLP-small: none vs perturbation
    oulad_mlp_none = find_runs(plan, "OULAD", "MLP", "none", publish_defense=None)
    oulad_mlp_pert = find_runs(plan, "OULAD", "MLP", "none", publish_defense="output_perturbation",
                              noise_type="gaussian", noise_scale=0.1)
    # Filter for MLP-small (variant=None or "small")
    oulad_mlp_none = [r for r in oulad_mlp_none if r.get("model_variant") in [None, "small"]]
    oulad_mlp_pert = [r for r in oulad_mlp_pert if r.get("model_variant") in [None, "small"]]
    
    print(f"Found {len(oulad_lr_none)} OULAD LR none runs")
    print(f"Found {len(oulad_lr_pert)} OULAD LR perturbation runs")
    print(f"Found {len(oulad_xgb_none)} OULAD XGBoost none runs")
    print(f"Found {len(oulad_xgb_pert)} OULAD XGBoost perturbation runs")
    print(f"Found {len(oulad_mlp_none)} OULAD MLP-small none runs")
    print(f"Found {len(oulad_mlp_pert)} OULAD MLP-small perturbation runs")
    print()
    
    # Check pairs (use seed=1 for consistency)
    pairs_to_check = []
    
    # Find matching seed pairs
    for none_run in oulad_lr_none[:5]:
        seed = none_run.get("seed", 1)
        pert_run = next((r for r in oulad_lr_pert if r.get("seed") == seed), None)
        if pert_run:
            pairs_to_check.append(("OULAD LR", none_run, pert_run))
            break
    
    for none_run in oulad_xgb_none[:5]:
        seed = none_run.get("seed", 1)
        pert_run = next((r for r in oulad_xgb_pert if r.get("seed") == seed), None)
        if pert_run:
            pairs_to_check.append(("OULAD XGBoost", none_run, pert_run))
            break
    
    for none_run in oulad_mlp_none[:5]:
        seed = none_run.get("seed", 1)
        pert_run = next((r for r in oulad_mlp_pert if r.get("seed") == seed), None)
        if pert_run:
            pairs_to_check.append(("OULAD MLP-small", none_run, pert_run))
            break
    
    print("Checking representative pairs:")
    print()
    
    all_results = []
    
    for label, base_run, pert_run in pairs_to_check:
        print(f"{label}:")
        print(f"  Base run: {base_run['run_id']} (seed={base_run.get('seed', 1)})")
        print(f"  Pert run: {pert_run['run_id']} (seed={pert_run.get('seed', 1)})")
        
        results = check_run_pair(base_run, pert_run, base_dir)
        all_results.append((label, results))
        
        print(f"  Base run - AUC from predictions_base.npy: {results['base_auc_base']:.4f}" if results['base_auc_base'] else "  Base run - AUC from predictions_base.npy: N/A")
        print(f"  Base run - AUC from predictions_released.npy: {results['base_auc_released']:.4f}" if results['base_auc_released'] else "  Base run - AUC from predictions_released.npy: N/A")
        print(f"  Pert run - AUC from predictions_base.npy: {results['pert_auc_base']:.4f}" if results['pert_auc_base'] else "  Pert run - AUC from predictions_base.npy: N/A")
        print(f"  Pert run - AUC from predictions_released.npy: {results['pert_auc_released']:.4f}" if results['pert_auc_released'] else "  Pert run - AUC from predictions_released.npy: N/A")
        print(f"  Base run - Rank correlation (base vs released): {results['base_rank_corr']:.4f}" if results['base_rank_corr'] is not None else "  Base run - Rank correlation: N/A")
        print(f"  Pert run - Rank correlation (base vs released): {results['pert_rank_corr']:.4f}" if results['pert_rank_corr'] is not None else "  Pert run - Rank correlation: N/A")
        print()
    
    # Summary table
    print("=" * 80)
    print("Summary Table: Base vs Released AUC")
    print("=" * 80)
    print(f"{'Model':<20} {'Base AUC (base)':<18} {'Base AUC (released)':<20} {'Pert AUC (base)':<18} {'Pert AUC (released)':<20} {'AUC Change':<12}")
    print("-" * 80)
    
    for label, results in all_results:
        base_auc_base = results['base_auc_base']
        base_auc_rel = results['base_auc_released']
        pert_auc_base = results['pert_auc_base']
        pert_auc_rel = results['pert_auc_released']
        
        auc_change = None
        if pert_auc_rel is not None and base_auc_base is not None:
            auc_change = pert_auc_rel - base_auc_base
        
        base_auc_base_str = f"{base_auc_base:.4f}" if base_auc_base else "N/A"
        base_auc_rel_str = f"{base_auc_rel:.4f}" if base_auc_rel else "N/A"
        pert_auc_base_str = f"{pert_auc_base:.4f}" if pert_auc_base else "N/A"
        pert_auc_rel_str = f"{pert_auc_rel:.4f}" if pert_auc_rel else "N/A"
        auc_change_str = f"{auc_change:+.4f}" if auc_change is not None else "N/A"
        
        print(f"{label:<20} {base_auc_base_str:<18} {base_auc_rel_str:<20} {pert_auc_base_str:<18} {pert_auc_rel_str:<20} {auc_change_str:<12}")
    
    print()
    print("Expected: Perturbation should slightly LOWER AUC (not increase)")
    print("If AUC increases dramatically, there's a bug in reporting/IO")

if __name__ == "__main__":
    main()
