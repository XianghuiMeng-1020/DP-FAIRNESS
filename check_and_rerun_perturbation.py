"""检查并重新运行需要更新的perturbation runs"""
import json
from pathlib import Path

plan = json.load(open('outputs/reports/experiment_plan_fast.json'))
pert_runs = [e for e in plan if e.get('publish_defense') == 'output_perturbation']

needs_rerun = []
for entry in pert_runs:
    run_id = entry['run_id']
    config_path = Path(f'outputs/runs/{run_id}/config.json')
    
    if not config_path.exists():
        continue
    
    cfg = json.load(open(config_path))
    # 检查config是否缺少perturbation配置
    if cfg.get('noise_type') is None or cfg.get('noise_scale') is None:
        needs_rerun.append(run_id)

print(f'Total perturbation runs: {len(pert_runs)}')
print(f'Need to rerun: {len(needs_rerun)}')
if needs_rerun:
    print(f'First 10: {needs_rerun[:10]}')
