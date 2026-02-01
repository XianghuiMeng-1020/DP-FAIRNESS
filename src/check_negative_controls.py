"""检查负控制runs是否被正确加载"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from reporting import load_plan, aggregate_metrics

plan = load_plan('outputs/reports/experiment_plan_fast.json')
agg = aggregate_metrics(plan, 'outputs/runs', False)

neg_runs = []
for key, runs in agg.items():
    for run in runs:
        if run.get('negative_control'):
            neg_runs.append(run)

print(f'Found {len(neg_runs)} negative control runs in aggregated data')
if neg_runs:
    print(f'Sample run_id: {neg_runs[0]["run_id"]}')
    print(f'Sample negative_control: {neg_runs[0].get("negative_control")}')
    print(f'Sample test_auc: {neg_runs[0].get("test_auc")}')

# 检查metrics.json是否存在
from pathlib import Path
sample_run = neg_runs[0]["run_id"] if neg_runs else None
if sample_run:
    metrics_path = Path(f'outputs/runs/{sample_run}/metrics.json')
    print(f'\nMetrics file exists: {metrics_path.exists()}')
    if metrics_path.exists():
        with open(metrics_path) as f:
            m = json.load(f)
            print(f'Test AUC in metrics.json: {m.get("test_auc")}')
            print(f'Negative control in metrics.json: {m.get("negative_control", "NOT FOUND")}')
