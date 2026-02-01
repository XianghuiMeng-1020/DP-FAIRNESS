"""Check which perturbation runs need to be rerun"""
import json
from pathlib import Path

plan_path = Path("outputs/reports/experiment_plan_fast.json")
with open(plan_path, "r") as f:
    plan = json.load(f)

perturbation_runs = [r for r in plan if r.get("publish_defense") == "output_perturbation"]
coarsening_runs = [r for r in plan if r.get("publish_defense") == "output_coarsening"]

print(f"Total perturbation runs: {len(perturbation_runs)}")
print(f"Total coarsening runs: {len(coarsening_runs)}")

# Check first 10 perturbation runs
print("\nFirst 10 perturbation runs:")
for r in perturbation_runs[:10]:
    run_id = r["run_id"]
    base_path = Path(f"outputs/runs/{run_id}/predictions_base.npy")
    released_path = Path(f"outputs/runs/{run_id}/predictions_released.npy")
    print(f"  {run_id}: base={base_path.exists()}, released={released_path.exists()}")

# Count how many need rerun
needs_rerun = []
for r in perturbation_runs + coarsening_runs:
    run_id = r["run_id"]
    base_path = Path(f"outputs/runs/{run_id}/predictions_base.npy")
    if not base_path.exists():
        needs_rerun.append(run_id)

print(f"\nRuns that need rerun: {len(needs_rerun)}")
if len(needs_rerun) > 0:
    print(f"First 10: {needs_rerun[:10]}")
