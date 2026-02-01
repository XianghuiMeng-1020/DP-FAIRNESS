"""
Fix negative control run_ids: rename directories to match plan
OR update plan to match actual directory names
"""
import json
from pathlib import Path
import shutil

def load_plan(plan_path="outputs/reports/experiment_plan_fast.json"):
    with open(plan_path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    plan = load_plan()
    base_dir = Path("outputs/runs")
    
    # Find negative control entries
    nc_entries = [e for e in plan if e.get("negative_control") is not None or "negative_control" in e.get("run_id", "")]
    
    print(f"Found {len(nc_entries)} negative control entries in plan")
    
    # Check actual directories
    actual_dirs = {}
    for nc_entry in nc_entries:
        plan_run_id = nc_entry["run_id"]
        # Check if directory exists with exact name
        if (base_dir / plan_run_id).exists():
            actual_dirs[plan_run_id] = plan_run_id
            continue
        
        # Check if directory exists with N instead of N/A
        alt_run_id = plan_run_id.replace("_N/A_", "_N_")
        if (base_dir / alt_run_id).exists():
            actual_dirs[plan_run_id] = alt_run_id
            print(f"Found: {plan_run_id} -> {alt_run_id}")
    
    print(f"\nMatching directories: {len(actual_dirs)}/{len(nc_entries)}")
    
    # Option: Update plan to use actual directory names
    # This is safer than renaming directories
    updated_plan = []
    for entry in plan:
        if entry["run_id"] in actual_dirs:
            entry["run_id"] = actual_dirs[entry["run_id"]]
        updated_plan.append(entry)
    
    # Save updated plan
    output_path = Path("outputs/reports/experiment_plan_fast.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(updated_plan, f, indent=2, ensure_ascii=False)
    
    print(f"\nUpdated plan saved to {output_path}")
    print("Run IDs with N/A have been changed to N to match actual directories")

if __name__ == "__main__":
    main()
