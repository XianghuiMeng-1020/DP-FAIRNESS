"""获取代表性run_id列表"""
import json
from pathlib import Path

plan = json.load(open("outputs/reports/experiment_plan_fast.json"))
runs_dir = Path("outputs/runs")

sample_runs = []
datasets = ["OULAD", "UCI697", "HarvardX_PersonCourse"]

for d in datasets:
    found = [e for e in plan if e["dataset"] == d]
    # 选择不同防御的代表性runs
    for entry in found[:2]:
        if entry["run_id"] not in sample_runs:
            sample_runs.append(entry["run_id"])

print("Representative run_id paths:")
for run_id in sample_runs[:6]:
    run_path = runs_dir / run_id
    print(f"{run_id}: {run_path}")
