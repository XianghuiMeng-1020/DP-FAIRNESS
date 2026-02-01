"""验证perturbation是否正确应用"""
import numpy as np
import json
from pathlib import Path

runs = ['fast_0005', 'fast_0246', 'fast_0300']
for rid in runs:
    cfg_path = Path(f'outputs/runs/{rid}/config.json')
    preds_path = Path(f'outputs/runs/{rid}/predictions.npy')
    
    if not cfg_path.exists() or not preds_path.exists():
        print(f'{rid}: Missing files')
        continue
    
    cfg = json.load(open(cfg_path))
    preds = np.load(preds_path)
    y_scores = preds[:, 1]
    
    print(f'{rid}:')
    print(f'  noise_type={cfg.get("noise_type")}')
    print(f'  noise_scale={cfg.get("noise_scale")}')
    print(f'  y_scores: mean={y_scores.mean():.4f}, std={y_scores.std():.4f}, min={y_scores.min():.4f}, max={y_scores.max():.4f}')
    print(f'  unique values: {len(np.unique(y_scores))}')
    print()
