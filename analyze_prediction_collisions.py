"""分析predictions.npy碰撞的原因"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

plan = json.load(open('outputs/reports/experiment_plan_fast.json'))

# 找到predictions.npy的碰撞
hash_to_runs = defaultdict(list)
for entry in plan:
    run_id = entry['run_id']
    manifest_path = Path(f'outputs/runs/{run_id}/artifact_manifest.json')
    if not manifest_path.exists():
        continue
    manifest = json.load(open(manifest_path))
    for artifact in manifest.get('artifacts', []):
        if artifact.get('file') == 'predictions.npy' and artifact.get('hash'):
            hash_to_runs[artifact['hash']].append(run_id)

collisions = {h: runs for h, runs in hash_to_runs.items() if len(runs) > 1}

print(f"Found {len(collisions)} prediction hash collisions")
print("\nAnalyzing collisions...")

for h, runs in list(collisions.items())[:3]:
    print(f"\nHash {h[:16]}... ({len(runs)} runs):")
    # 检查这些runs的配置
    configs = []
    for run_id in runs[:10]:
        cfg_path = Path(f'outputs/runs/{run_id}/config.json')
        if cfg_path.exists():
            cfg = json.load(open(cfg_path))
            configs.append({
                'run_id': run_id,
                'dataset': cfg.get('dataset'),
                'model': cfg.get('model'),
                'seed': cfg.get('seed'),
                'train_def': cfg.get('train_defense'),
                'pub_def': cfg.get('publish_defense'),
                'eps': cfg.get('eps'),
            })
    
    # 检查是否所有配置都相同（除了run_id）
    if len(configs) > 1:
        first = configs[0]
        all_same = all(
            c['dataset'] == first['dataset'] and
            c['model'] == first['model'] and
            c['seed'] == first['seed'] and
            c['train_def'] == first['train_def'] and
            c['pub_def'] == first['pub_def'] and
            c['eps'] == first['eps']
            for c in configs[1:]
        )
        
        if all_same:
            print(f"  [EXPECTED] All runs have identical config (same seed/model/defense)")
            print(f"    Example: {configs[0]}")
        else:
            print(f"  [UNEXPECTED] Runs have different configs!")
            for c in configs[:5]:
                print(f"    {c}")

# 检查是否有不同seed但相同hash的情况（这是bug）
print("\n" + "="*60)
print("Checking for bug: same hash with different seeds")
print("="*60)

bug_found = False
for h, runs in collisions.items():
    seeds = set()
    for run_id in runs:
        cfg_path = Path(f'outputs/runs/{run_id}/config.json')
        if cfg_path.exists():
            cfg = json.load(open(cfg_path))
            seeds.add(cfg.get('seed'))
    
    if len(seeds) > 1:
        print(f"[BUG] Hash {h[:16]}... has runs with different seeds: {seeds}")
        bug_found = True
        # 检查实际predictions是否真的相同
        sample_runs = runs[:2]
        preds_list = []
        for run_id in sample_runs:
            pred_path = Path(f'outputs/runs/{run_id}/predictions.npy')
            if pred_path.exists():
                preds_list.append((run_id, np.load(pred_path)))
        
        if len(preds_list) == 2:
            r1, p1 = preds_list[0]
            r2, p2 = preds_list[1]
            if np.array_equal(p1, p2):
                print(f"  Confirmed: {r1} and {r2} have identical predictions (BUG!)")
            else:
                print(f"  Note: {r1} and {r2} have different predictions but same hash (hash collision in hash function)")

if not bug_found:
    print("[PASS] No bug found - collisions are expected for runs with identical configs")
