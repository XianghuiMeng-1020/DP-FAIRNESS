"""
Quick progress checker - run this to see current status
"""
import sys
import io
from pathlib import Path
import json
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def check_progress():
    """Check current experiment progress"""
    plan_path = Path("outputs/reports/experiment_plan_fast.json")
    runs_dir = Path("outputs/runs")
    
    if not plan_path.exists():
        print("Plan file not found!")
        return
    
    plan = json.load(open(plan_path, encoding='utf-8'))
    total_runs = len(plan)
    
    completed = 0
    failed = 0
    pending = 0
    
    for entry in plan:
        run_id = entry["run_id"]
        run_dir = runs_dir / run_id.replace("N/A", "N")
        status_file = run_dir / "status.json"
        
        if status_file.exists():
            try:
                status = json.load(open(status_file, encoding='utf-8'))
                if status.get("status") == "ok":
                    completed += 1
                elif status.get("status") == "failed":
                    failed += 1
                else:
                    pending += 1
            except:
                pending += 1
        else:
            pending += 1
    
    print("="*60)
    print(f"Experiment Progress - {datetime.now().strftime('%H:%M:%S')}")
    print("="*60)
    print(f"Total runs: {total_runs}")
    print(f"Completed: {completed} ({completed*100/total_runs:.1f}%)")
    print(f"Failed: {failed}")
    print(f"Pending: {pending}")
    print("="*60)
    
    if completed > 0:
        filled = int(completed * 50 / total_runs)
        empty = 50 - filled
        progress_bar = "#" * filled + "." * empty
        print(f"[{progress_bar}] {completed*100/total_runs:.1f}%")

if __name__ == "__main__":
    check_progress()
