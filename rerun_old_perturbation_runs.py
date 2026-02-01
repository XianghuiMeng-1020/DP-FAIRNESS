"""重新运行旧的perturbation runs（在修复之前运行的）"""
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
        needs_rerun.append(entry)

print(f'Need to rerun: {len(needs_rerun)} runs')

# 更新config.json以包含正确的perturbation配置，并删除predictions以强制重新生成
for entry in needs_rerun:
    run_id = entry['run_id']
    run_dir = Path(f'outputs/runs/{run_id}')
    config_path = run_dir / 'config.json'
    
    if config_path.exists():
        cfg = json.load(open(config_path))
        cfg['noise_type'] = entry.get('noise_type')
        cfg['noise_scale'] = entry.get('noise_scale')
        cfg['publish_defense'] = entry.get('publish_defense')
        json.dump(cfg, open(config_path, 'w'), indent=2, ensure_ascii=False)
        
        # 删除predictions.npy以强制重新生成（应用perturbation）
        preds_path = run_dir / 'predictions.npy'
        if preds_path.exists():
            try:
                preds_path.unlink()
            except:
                pass  # 忽略权限错误

print(f'Updated {len(needs_rerun)} runs')
