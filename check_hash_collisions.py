"""检查hash碰撞详情"""
import json
from pathlib import Path
from collections import defaultdict

hash_to_runs = defaultdict(list)
plan = json.load(open('outputs/reports/experiment_plan_fast.json'))

for entry in plan:
    run_id = entry['run_id']
    manifest_path = Path(f'outputs/runs/{run_id}/artifact_manifest.json')
    if not manifest_path.exists():
        continue
    
    manifest = json.load(open(manifest_path))
    for artifact in manifest.get('artifacts', []):
        if artifact.get('hash'):
            hash_to_runs[artifact['hash']].append((run_id, artifact.get('file')))

collisions = {h: runs for h, runs in hash_to_runs.items() if len(runs) > 1}

print(f"Total unique hashes: {len(hash_to_runs)}")
print(f"Hash collisions: {len(collisions)}")
print("\nCollision details:")

# 按文件类型分组
collisions_by_file = defaultdict(list)
for h, runs in collisions.items():
    file_types = {}
    for run_id, filename in runs:
        file_types[filename] = file_types.get(filename, 0) + 1
    collisions_by_file[list(file_types.keys())[0] if file_types else 'unknown'].append((h, runs))

for file_type, coll_list in collisions_by_file.items():
    print(f"\n{file_type}: {len(coll_list)} collisions")
    for h, runs in coll_list[:3]:
        print(f"  Hash {h[:16]}...: {len(runs)} runs")
        if file_type == 'predictions.npy':
            print(f"    Example runs: {[r[0] for r in runs[:5]]}")

# 检查predictions.npy的碰撞（这是最重要的）
pred_collisions = [runs for h, runs in collisions.items() 
                   if any(f == 'predictions.npy' for _, f in runs)]
if pred_collisions:
    print(f"\n[WARNING] Found {len(pred_collisions)} predictions.npy collisions!")
else:
    print(f"\n[PASS] No predictions.npy collisions (other collisions are expected for shared files like test_labels.npy)")
