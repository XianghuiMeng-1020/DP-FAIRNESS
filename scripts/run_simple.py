"""
Simple script to run remaining experiments with visible output
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.run_all import load_plan, run_experiment, get_run_dir

# Load plan
plan = load_plan("outputs/reports/experiment_plan_fast.json")
total = len(plan)

print(f"Total runs: {total}")
print("="*60)

# Find pending runs
pending = []
for entry in plan:
    run_dir = get_run_dir(entry["run_id"])
    if not (run_dir / "status.json").exists():
        pending.append(entry)

print(f"Pending runs: {len(pending)}")
print("="*60)
print()

# Run each pending run
for i, entry in enumerate(pending):
    run_id = entry["run_id"]
    print(f"[{i+1}/{len(pending)}] Running {run_id}...")
    print(f"  Dataset: {entry.get('dataset')}")
    print(f"  Model: {entry.get('model')}")
    
    try:
        result = run_experiment(entry, base_dir="outputs/runs")
        if result["status"] == "ok":
            print(f"  [OK] Completed")
        else:
            print(f"  [FAILED] {result.get('error', 'Unknown')[:100]}")
    except Exception as e:
        print(f"  [ERROR] {str(e)[:100]}")
    
    print()

print("="*60)
print("Done!")
