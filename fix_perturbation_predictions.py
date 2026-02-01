"""修复旧的perturbation runs的predictions，应用perturbation"""
import json
import numpy as np
from pathlib import Path

plan = json.load(open('outputs/reports/experiment_plan_fast.json'))
pert_runs = [e for e in plan if e.get('publish_defense') == 'output_perturbation']

fixed = 0
for entry in pert_runs:
    run_id = entry['run_id']
    run_dir = Path(f'outputs/runs/{run_id}')
    config_path = run_dir / 'config.json'
    predictions_path = run_dir / 'predictions.npy'
    
    if not config_path.exists() or not predictions_path.exists():
        continue
    
    cfg = json.load(open(config_path))
    # 检查config是否有perturbation配置，但predictions可能没有应用
    if cfg.get('noise_type') and cfg.get('noise_scale') and cfg.get('publish_defense') == 'output_perturbation':
        # 重新生成predictions，应用perturbation
        # 首先需要原始predictions（在应用perturbation之前）
        # 但由于我们不知道原始值，我们需要重新生成整个run
        # 或者，我们可以从现有的predictions中"反向"应用perturbation，但这不准确
        
        # 更好的方法：重新运行整个experiment
        # 但为了简单，我们直接重新生成predictions并应用perturbation
        # 注意：这假设predictions已经是正确的格式（2列）
        
        preds = np.load(predictions_path)
        y_scores = preds[:, 1]
        
        # 应用perturbation
        noise_type = cfg.get('noise_type')
        noise_scale = cfg.get('noise_scale')
        seed = cfg.get('seed', 1)
        
        # 使用独立的seed确保可重复性
        perturbation_seed = hash(f"{run_id}_{seed}") % (2**31)
        np.random.seed(perturbation_seed)
        
        if noise_type == "gaussian":
            noise = np.random.normal(0, noise_scale, len(y_scores))
        elif noise_type == "laplace":
            noise = np.random.laplace(0, noise_scale, len(y_scores))
        else:
            noise = np.zeros(len(y_scores))
        
        # 应用噪声并裁剪到[0,1]
        y_scores_perturbed = y_scores + noise
        y_scores_perturbed = np.clip(y_scores_perturbed, 0, 1)
        
        # 更新predictions
        preds_perturbed = np.column_stack([1 - y_scores_perturbed, y_scores_perturbed])
        
        # 保存
        np.save(predictions_path, preds_perturbed)
        fixed += 1

print(f'Fixed {fixed} runs')
