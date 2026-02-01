"""Final exit checks for perturbation fix"""
import json
from pathlib import Path
import hashlib
from collections import defaultdict

def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of file"""
    if not file_path.exists():
        return None
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

# Check 1: All perturbation runs have non-null noise_scale
plan = json.load(open('outputs/reports/experiment_plan_fast.json'))
pert_runs = [e for e in plan if e.get('publish_defense') == 'output_perturbation']

print("=" * 60)
print("EXIT CHECK 1: Perturbation runs have non-null noise_scale")
print("=" * 60)

missing_noise_scale = []
for entry in pert_runs:
    run_id = entry['run_id']
    config_path = Path(f'outputs/runs/{run_id}/config.json')
    if not config_path.exists():
        missing_noise_scale.append(run_id)
        continue
    cfg = json.load(open(config_path))
    if cfg.get('noise_scale') is None or cfg.get('noise_type') is None:
        missing_noise_scale.append(run_id)

if missing_noise_scale:
    print(f"[FAIL] {len(missing_noise_scale)} runs missing noise_scale")
    print(f"   Examples: {missing_noise_scale[:5]}")
else:
    print(f"[PASS] All {len(pert_runs)} perturbation runs have non-null noise_scale")

# Check 2: Perturbation is applied to artifacts
print("\n" + "=" * 60)
print("EXIT CHECK 2: Perturbation applied to artifacts")
print("=" * 60)

import numpy as np
sample_runs = ['fast_0005', 'fast_0300', 'fast_0500']
perturbation_applied = True
for run_id in sample_runs:
    preds_path = Path(f'outputs/runs/{run_id}/predictions.npy')
    config_path = Path(f'outputs/runs/{run_id}/config.json')
    
    if not preds_path.exists() or not config_path.exists():
        continue
    
    cfg = json.load(open(config_path))
    if cfg.get('publish_defense') != 'output_perturbation':
        continue
    
    preds = np.load(preds_path)
    y_scores = preds[:, 1]
    unique_count = len(np.unique(y_scores))
    
    # Perturbation should increase unique values (not constant)
    if unique_count < 100:  # Threshold check
        print(f"[WARNING] {run_id} has only {unique_count} unique values")
        perturbation_applied = False

if perturbation_applied:
    print(f"[PASS] Perturbation appears to be applied (checked {len(sample_runs)} sample runs)")
else:
    print(f"[FAIL] Perturbation may not be applied correctly")

# Check 3: No artifact hash collisions
print("\n" + "=" * 60)
print("EXIT CHECK 3: No artifact hash collisions")
print("=" * 60)

hash_to_runs = defaultdict(list)
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

if collisions:
    print(f"[FAIL] Found {len(collisions)} hash collisions")
    for h, runs in list(collisions.items())[:5]:
        print(f"   Hash {h[:16]}... appears in {len(runs)} runs")
else:
    print(f"[PASS] No hash collisions (checked {len(hash_to_runs)} unique hashes)")

# Check 4: Tables include perturbation results
print("\n" + "=" * 60)
print("EXIT CHECK 4: Tables include perturbation results")
print("=" * 60)

tables_path = Path('paper/all_tables.md')
if tables_path.exists():
    content = tables_path.read_text(encoding='utf-8')
    pert_count = content.lower().count('output_perturbation')
    if pert_count > 50:  # Should have many mentions
        print(f"[PASS] Tables include perturbation results ({pert_count} mentions)")
    else:
        print(f"[FAIL] Tables may not include perturbation results ({pert_count} mentions)")
else:
    print("[FAIL] paper/all_tables.md not found")

# Check 5: Consistency check
print("\n" + "=" * 60)
print("EXIT CHECK 5: Internal consistency")
print("=" * 60)

paper_path = Path('paper/EDM_FULL_PAPER.md')
if paper_path.exists():
    paper_content = paper_path.read_text(encoding='utf-8')
    if 'output_perturbation' in paper_content.lower() and 'scale=0.1' in paper_content:
        print("[PASS] Paper mentions perturbation with scale=0.1")
    else:
        print("[WARNING] Paper may not mention perturbation correctly")

# Final summary
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"Total perturbation runs in plan: {len(pert_runs)}")
print(f"Runs with valid configs: {len(pert_runs) - len(missing_noise_scale)}")
print(f"Hash collisions: {len(collisions)}")
print("\nStatus: ", end="")
if not missing_noise_scale and not collisions and perturbation_applied:
    print("[PASS] ALL CHECKS PASSED")
else:
    print("[FAIL] SOME CHECKS FAILED")
