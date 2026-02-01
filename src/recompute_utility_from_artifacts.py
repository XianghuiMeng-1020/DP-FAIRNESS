"""重新从artifacts计算utility metrics（test_auc, test_f1, ece），使用正确的文件规则"""
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score
from typing import Dict, Any, Optional

def recompute_test_auc_from_correct_file(run_dir: Path, publish_defense: Optional[str]) -> Optional[float]:
    """根据release defense规则，从正确的文件重新计算test_auc"""
    labels_path = run_dir / "labels.npy"
    if not labels_path.exists():
        return None
    
    labels = np.load(labels_path)
    
    # 规则：如果release defense是none，使用predictions_base；否则使用predictions_released
    if publish_defense in ["output_coarsening", "output_perturbation"]:
        predictions_path = run_dir / "predictions_released.npy"
    else:
        predictions_path = run_dir / "predictions_base.npy"
    
    if not predictions_path.exists():
        return None
    
    predictions = np.load(predictions_path)
    
    # 提取正类概率
    if len(predictions.shape) > 1 and predictions.shape[1] == 2:
        y_scores = predictions[:, 1]
    else:
        y_scores = predictions.flatten()
    
    y_true = labels.flatten()
    
    if len(np.unique(y_true)) != 2:
        return None
    
    try:
        return roc_auc_score(y_true, y_scores)
    except:
        return None

def recompute_test_f1_from_correct_file(run_dir: Path, publish_defense: Optional[str]) -> Optional[float]:
    """根据release defense规则，从正确的文件重新计算test_f1"""
    labels_path = run_dir / "labels.npy"
    if not labels_path.exists():
        return None
    
    labels = np.load(labels_path)
    
    # 规则：如果release defense是none，使用predictions_base；否则使用predictions_released
    if publish_defense in ["output_coarsening", "output_perturbation"]:
        predictions_path = run_dir / "predictions_released.npy"
    else:
        predictions_path = run_dir / "predictions_base.npy"
    
    if not predictions_path.exists():
        return None
    
    predictions = np.load(predictions_path)
    
    # 提取正类概率并转换为二分类预测
    if len(predictions.shape) > 1 and predictions.shape[1] == 2:
        y_proba = predictions[:, 1]
    else:
        y_proba = predictions.flatten()
    
    y_pred = (y_proba >= 0.5).astype(int)
    y_true = labels.flatten()
    
    if len(np.unique(y_true)) != 2:
        return None
    
    try:
        return f1_score(y_true, y_pred)
    except:
        return None

def recompute_ece_from_correct_file(run_dir: Path, publish_defense: Optional[str], n_bins: int = 10) -> Optional[float]:
    """根据release defense规则，从正确的文件重新计算ECE"""
    labels_path = run_dir / "labels.npy"
    if not labels_path.exists():
        return None
    
    labels = np.load(labels_path)
    
    # 规则：如果release defense是none，使用predictions_base；否则使用predictions_released
    if publish_defense in ["output_coarsening", "output_perturbation"]:
        predictions_path = run_dir / "predictions_released.npy"
    else:
        predictions_path = run_dir / "predictions_base.npy"
    
    if not predictions_path.exists():
        return None
    
    predictions = np.load(predictions_path)
    
    # 提取正类概率
    if len(predictions.shape) > 1 and predictions.shape[1] == 2:
        y_proba = predictions[:, 1]
    else:
        y_proba = predictions.flatten()
    
    y_true = labels.flatten()
    
    if len(np.unique(y_true)) != 2:
        return None
    
    try:
        # ECE计算：将概率分成n_bins，计算每个bin的校准误差
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # 找到在这个bin中的样本
            in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                # bin内的平均预测概率
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_proba[in_bin].mean()
                # 加权校准误差
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return float(ece)
    except:
        return None

def recompute_utility_metrics_for_run(run_dir: Path, publish_defense: Optional[str]) -> Dict[str, Any]:
    """为单个run重新计算所有utility metrics"""
    result = {}
    
    test_auc = recompute_test_auc_from_correct_file(run_dir, publish_defense)
    if test_auc is not None:
        result["test_auc"] = test_auc
    
    test_f1 = recompute_test_f1_from_correct_file(run_dir, publish_defense)
    if test_f1 is not None:
        result["test_f1"] = test_f1
    
    ece = recompute_ece_from_correct_file(run_dir, publish_defense)
    if ece is not None:
        result["ece"] = ece
    
    return result

def update_all_metrics_json(base_dir: str = "outputs/runs", plan_path: str = "outputs/reports/experiment_plan_fast.json"):
    """更新所有runs的metrics.json，使用正确的文件重新计算utility metrics"""
    import json
    from reporting import load_plan
    
    plan = load_plan(plan_path)
    base_path = Path(base_dir)
    
    updated_count = 0
    error_count = 0
    
    for entry in plan:
        run_id = entry["run_id"]
        run_dir = base_path / run_id
        
        if not run_dir.exists():
            continue
        
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        
        # 加载现有metrics
        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
        except:
            error_count += 1
            continue
        
        # 获取publish_defense
        publish_defense = entry.get("publish_defense") or "none"
        if publish_defense == "none":
            publish_defense = None
        
        # 重新计算utility metrics
        recomputed = recompute_utility_metrics_for_run(run_dir, publish_defense)
        
        # 更新metrics
        metrics.update(recomputed)
        
        # 保存
        try:
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            updated_count += 1
        except:
            error_count += 1
    
    print(f"Updated {updated_count} runs, {error_count} errors")
    return updated_count, error_count

if __name__ == "__main__":
    update_all_metrics_json()
