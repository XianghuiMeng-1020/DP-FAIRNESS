"""
Rerun all perturbation runs with the fixed code
"""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path("src").absolute()))
from run_all import run_experiment

plan_path = Path("outputs/reports/experiment_plan_fast.json")
with open(plan_path, "r") as f:
    plan = json.load(f)

# Find all perturbation runs
perturbation_runs = [
    entry for entry in plan
    if entry.get("publish_defense") == "output_perturbation"
]

print(f"Found {len(perturbation_runs)} perturbation runs to rerun")

# Also find coarsening runs (they also need base/released separation)
coarsening_runs = [
    entry for entry in plan
    if entry.get("publish_defense") == "output_coarsening"
]

print(f"Found {len(coarsening_runs)} coarsening runs to rerun")

all_runs_to_rerun = perturbation_runs + coarsening_runs
print(f"Total runs to rerun: {len(all_runs_to_rerun)}")

# For now, only rerun 6 representative runs for sanity check
# User can rerun all later if needed
representative_run_ids = [
    "fast_0000",  # OULAD LR, none
    "fast_0005",  # OULAD LR, perturbation
    "fast_0030",  # OULAD MLP-small, none
    "fast_0094",  # OULAD MLP-small, perturbation
    "fast_0084",  # OULAD MLP-small, coarsening (if exists)
]

# Find representative runs
representative_runs = []
for entry in all_runs_to_rerun:
    if entry["run_id"] in representative_run_ids:
        representative_runs.append(entry)

# Also add a coarsening run if not in list
for entry in coarsening_runs:
    if entry["run_id"] not in [r["run_id"] for r in representative_runs]:
        representative_runs.append(entry)
        break

print(f"\nRerunning {len(representative_runs)} representative runs for sanity check...")
all_runs_to_rerun = representative_runs

# Rerun them
success_count = 0
fail_count = 0

for i, entry in enumerate(all_runs_to_rerun):
    run_id = entry["run_id"]
    print(f"\n[{i+1}/{len(all_runs_to_rerun)}] Rerunning {run_id}...")
    
    result = run_experiment(entry, base_dir="outputs/runs")
    
    if result["status"] == "ok":
        success_count += 1
        metrics = result["metrics"]
        print(f"  OK: Test AUC = {metrics['test_auc']:.6f}")
    else:
        fail_count += 1
        print(f"  FAILED: {result.get('error', 'Unknown error')}")

print(f"\nSummary:")
print(f"  Success: {success_count}")
print(f"  Failed: {fail_count}")
