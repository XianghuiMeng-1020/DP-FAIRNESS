"""
Generate missing negative control runs
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.run_all import run_experiment

def load_plan(plan_path="outputs/reports/experiment_plan_fast.json"):
    with open(plan_path, "r", encoding="utf-8") as f:
        return json.load(f)

def check_run_exists(run_id, base_dir="outputs/runs"):
    run_dir = Path(base_dir) / run_id
    metrics_path = run_dir / "metrics.json"
    base_pred_path = run_dir / "predictions_base.npy"
    released_pred_path = run_dir / "predictions_released.npy"
    return metrics_path.exists() and base_pred_path.exists() and released_pred_path.exists()

def main():
    plan = load_plan()
    base_dir = "outputs/runs"
    
    # Find all negative control entries
    negative_controls = []
    for entry in plan:
        if entry.get("negative_control") is not None:
            negative_controls.append(entry)
        elif "negative_control" in entry.get("run_id", "").lower():
            negative_controls.append(entry)
    
    print(f"Found {len(negative_controls)} negative control entries in plan")
    
    # Check which ones are missing
    missing = []
    existing = []
    
    for entry in negative_controls:
        run_id = entry["run_id"]
        if check_run_exists(run_id, base_dir):
            existing.append(run_id)
        else:
            missing.append(entry)
    
    print(f"Existing: {len(existing)}")
    print(f"Missing: {len(missing)}")
    print()
    
    if len(missing) == 0:
        print("All negative controls exist!")
        return
    
    print("Missing negative controls:")
    for entry in missing[:10]:  # Show first 10
        print(f"  {entry['run_id']}")
    if len(missing) > 10:
        print(f"  ... and {len(missing) - 10} more")
    print()
    
    # Generate missing runs
    print(f"Generating {len(missing)} missing negative control runs...")
    print()
    
    success_count = 0
    fail_count = 0
    
    for i, entry in enumerate(missing, 1):
        run_id = entry["run_id"]
        print(f"[{i}/{len(missing)}] Running {run_id}...")
        
        try:
            result = run_experiment(entry, base_dir=base_dir)
            if result.get("status") == "ok":
                success_count += 1
                print(f"  ✓ Success")
            else:
                fail_count += 1
                print(f"  ✗ Failed: {result.get('error', 'Unknown error')}")
        except Exception as e:
            fail_count += 1
            print(f"  ✗ Exception: {e}")
        print()
    
    print("=" * 80)
    print(f"Summary: {success_count} succeeded, {fail_count} failed")
    print("=" * 80)

if __name__ == "__main__":
    main()
