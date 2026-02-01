"""
Check why Table 6 shows wrong AUC values
"""
import json
from pathlib import Path

def load_metrics(run_id, base_dir="outputs/runs"):
    metrics_path = Path(base_dir) / run_id / "metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path, "r") as f:
        return json.load(f)

plan = json.load(open("outputs/reports/experiment_plan_fast.json", encoding="utf-8"))

# Find OULAD LR none runs
oulad_lr_none = [e for e in plan if e['dataset']=='OULAD' and e['model']=='LR' 
                 and e.get('train_defense')=='none' and e.get('publish_defense') is None]

print("OULAD LR none runs:")
print(f"Total: {len(oulad_lr_none)}")
print()

aucs_regular = []
aucs_negctrl = []

for entry in oulad_lr_none:
    run_id = entry['run_id']
    metrics = load_metrics(run_id)
    if metrics:
        auc = metrics.get('test_auc')
        is_negctrl = entry.get('negative_control') is not None or 'negative_control' in run_id.lower()
        print(f"{run_id}: AUC={auc:.4f}, negctrl={is_negctrl}")
        if is_negctrl:
            aucs_negctrl.append(auc)
        else:
            aucs_regular.append(auc)

print()
print(f"Regular runs ({len(aucs_regular)}): mean={sum(aucs_regular)/len(aucs_regular):.4f}")
print(f"Negative control runs ({len(aucs_negctrl)}): mean={sum(aucs_negctrl)/len(aucs_negctrl):.4f}")
print(f"All runs mean: {sum(aucs_regular + aucs_negctrl)/len(aucs_regular + aucs_negctrl):.4f}")
