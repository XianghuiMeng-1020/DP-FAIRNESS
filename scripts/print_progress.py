"""
Progress monitoring script for full grid rerun
Prints progress every 30 seconds: completed/total/recent run_id/failed count
"""
import json
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def count_completed_runs(base_dir="outputs/runs"):
    """Count completed runs (have predictions_base.npy, predictions_released.npy, and metrics.json)"""
    runs_dir = Path(base_dir)
    if not runs_dir.exists():
        return 0, []
    
    completed = []
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        
        base_pred = run_dir / "predictions_base.npy"
        released_pred = run_dir / "predictions_released.npy"
        metrics = run_dir / "metrics.json"
        
        # 完成标准：必须同时存在 predictions_base.npy 和 predictions_released.npy 以及 metrics.json
        if base_pred.exists() and released_pred.exists() and metrics.exists():
            completed.append(run_dir)
    
    return len(completed), completed

def count_failed_runs(base_dir="outputs/runs"):
    """Count failed runs (have failure_record.json or status.json with status=failed)"""
    runs_dir = Path(base_dir)
    if not runs_dir.exists():
        return 0
    
    failed = 0
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        
        failure_record = run_dir / "failure_record.json"
        status_file = run_dir / "status.json"
        
        if failure_record.exists():
            failed += 1
        elif status_file.exists():
            try:
                with open(status_file, "r") as f:
                    status = json.load(f)
                if status.get("status") == "failed":
                    failed += 1
            except:
                pass
    
    return failed

def get_recent_run_id(base_dir="outputs/runs"):
    """Get most recently modified run_id"""
    runs_dir = Path(base_dir)
    if not runs_dir.exists():
        return None
    
    recent_run = None
    recent_time = 0
    
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        
        # Check modification time of metrics.json or config.json
        for check_file in ["metrics.json", "config.json", "predictions_base.npy"]:
            check_path = run_dir / check_file
            if check_path.exists():
                mtime = check_path.stat().st_mtime
                if mtime > recent_time:
                    recent_time = mtime
                    recent_run = run_dir.name
                    break
    
    return recent_run

def get_total_runs(plan_path="outputs/reports/experiment_plan_fast.json"):
    """Get total number of runs from plan"""
    plan_path = Path(plan_path)
    if not plan_path.exists():
        return None
    
    try:
        with open(plan_path, "r") as f:
            plan = json.load(f)
        return len(plan)
    except:
        return None

def main():
    """Print progress summary"""
    base_dir = "outputs/runs"
    plan_path = "outputs/reports/experiment_plan_fast.json"
    
    total = get_total_runs(plan_path)
    completed_count, completed_runs = count_completed_runs(base_dir)
    failed_count = count_failed_runs(base_dir)
    recent_run = get_recent_run_id(base_dir)
    
    print("=" * 80)
    print(f"Progress Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    if total is not None:
        print(f"Total runs: {total}")
        print(f"Completed: {completed_count} ({completed_count/total*100:.1f}%)")
        print(f"Failed: {failed_count}")
        print(f"Remaining: {total - completed_count - failed_count}")
    else:
        print(f"Completed: {completed_count}")
        print(f"Failed: {failed_count}")
    
    if recent_run:
        print(f"Recent run: {recent_run}")
    else:
        print("Recent run: None")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
