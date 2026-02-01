"""
Check if experiment is stuck
"""
import sys
import io
from pathlib import Path
import json
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def check_stuck():
    runs_dir = Path("outputs/runs")
    plan = json.load(open("outputs/reports/experiment_plan_fast.json", encoding='utf-8'))
    
    print("Checking if experiment is stuck...")
    print("="*70)
    
    # Find last completed run
    last_completed_time = None
    last_completed_id = None
    
    for entry in plan:
        run_id = entry["run_id"]
        run_dir = runs_dir / run_id.replace("N/A", "N")
        status_file = run_dir / "status.json"
        
        if status_file.exists():
            try:
                status = json.load(open(status_file, encoding='utf-8'))
                if status.get("status") == "ok":
                    mtime = status_file.stat().st_mtime
                    if last_completed_time is None or mtime > last_completed_time:
                        last_completed_time = mtime
                        last_completed_id = run_id
            except:
                pass
    
    if last_completed_time:
        last_time = datetime.fromtimestamp(last_completed_time)
        now = datetime.now()
        age_minutes = (now.timestamp() - last_completed_time) / 60
        
        print(f"Last completed run: {last_completed_id}")
        print(f"Completed at: {last_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Age: {age_minutes:.1f} minutes ago")
        
        if age_minutes > 30:
            print(f"\nWARNING: No runs completed in {age_minutes:.1f} minutes - likely stuck!")
            return True
        else:
            print(f"\nOK: Last run completed {age_minutes:.1f} minutes ago")
            return False
    else:
        print("No completed runs found")
        return True

if __name__ == "__main__":
    stuck = check_stuck()
    sys.exit(1 if stuck else 0)
