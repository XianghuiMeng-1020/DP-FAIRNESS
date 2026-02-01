"""诊断fairness gap问题"""
import numpy as np
import json
from pathlib import Path

run_id = "fast_0000"
run_dir = Path(f"outputs/runs/{run_id}")

preds = np.load(run_dir / "predictions.npy")
labels = np.load(run_dir / "test_labels.npy")
groups = np.load(run_dir / "groups.npy")
metrics = json.load(open(run_dir / "metrics.json"))

y_scores = preds[:, 1]
y_pred = (y_scores >= 0.5).astype(int)
y_true = labels.flatten()
groups_flat = groups.flatten()
unique_groups = np.unique(groups_flat)

print(f"Unique groups: {unique_groups}")
print(f"Group counts: {[np.sum(groups_flat==g) for g in unique_groups]}")
print(f"Label distribution: {[np.sum(y_true==l) for l in [0,1]]}")

group_tprs = []
for g in unique_groups:
    mask = (groups_flat == g)
    group_true = y_true[mask]
    group_pred = y_pred[mask]
    
    tp = np.sum((group_true == 1) & (group_pred == 1))
    fn = np.sum((group_true == 1) & (group_pred == 0))
    pos_count = np.sum(group_true == 1)
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    group_tprs.append(tpr)
    
    print(f"\nGroup {g}:")
    print(f"  Total samples: {np.sum(mask)}")
    print(f"  Positive samples: {pos_count}")
    print(f"  TP: {tp}, FN: {fn}")
    print(f"  TPR: {tpr}")

print(f"\nGroup TPRs: {group_tprs}")
print(f"TPR Gap (max-min): {max(group_tprs) - min(group_tprs)}")
print(f"Metrics worst_group_tpr_gap: {metrics.get('worst_group_tpr_gap')}")
