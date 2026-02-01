"""验证perturbation runs的AUC计算是否正确"""
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
import json

# 检查6个代表性runs
test_runs = [
    ("fast_0005", "OULAD", "LR", "none", "output_perturbation"),
    ("fast_0006", "OULAD", "LR", "none", "output_perturbation"),
    ("fast_0020", "OULAD", "XGBoost", "none", "output_perturbation"),
    ("fast_0021", "OULAD", "XGBoost", "none", "output_perturbation"),
    ("fast_0035", "OULAD", "MLP-small", "none", "output_perturbation"),
    ("fast_0036", "OULAD", "MLP-small", "none", "output_perturbation"),
]

base_dir = Path("outputs/runs")

print("Run ID | Dataset | Model | Base AUC | Released AUC | Diff | Metrics.json AUC")
print("-" * 90)

for run_id, dataset, model, train_def, pub_def in test_runs:
    run_dir = base_dir / run_id
    if not run_dir.exists():
        print(f"{run_id} | NOT FOUND")
        continue
    
    # 加载labels
    labels_path = run_dir / "test_labels.npy"
    if not labels_path.exists():
        labels_path = run_dir / "labels.npy"
    
    if not labels_path.exists():
        print(f"{run_id} | NO LABELS")
        continue
    
    labels = np.load(labels_path)
    
    # 加载predictions
    base_path = run_dir / "predictions_base.npy"
    released_path = run_dir / "predictions_released.npy"
    
    if not base_path.exists() or not released_path.exists():
        print(f"{run_id} | MISSING PREDICTIONS")
        continue
    
    base_pred = np.load(base_path)
    released_pred = np.load(released_path)
    
    # 提取正类概率
    if len(base_pred.shape) > 1 and base_pred.shape[1] == 2:
        base_scores = base_pred[:, 1]
        released_scores = released_pred[:, 1]
    else:
        base_scores = base_pred.flatten()
        released_scores = released_pred.flatten()
    
    # 计算AUC
    base_auc = roc_auc_score(labels, base_scores)
    released_auc = roc_auc_score(labels, released_scores)
    diff = base_auc - released_auc
    
    # 加载metrics.json
    metrics_path = run_dir / "metrics.json"
    metrics_auc = None
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
            metrics_auc = metrics.get("test_auc")
    
    metrics_str = f"{metrics_auc:.5f}" if metrics_auc is not None else "N/A"
    print(f"{run_id} | {dataset} | {model} | {base_auc:.5f} | {released_auc:.5f} | {diff:+.5f} | {metrics_str}")
