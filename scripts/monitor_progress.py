"""
Monitor experiment progress with detailed status
"""
import sys
import io
import time
from pathlib import Path
import json
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def monitor():
    plan = json.load(open("outputs/reports/experiment_plan_fast.json", encoding='utf-8'))
    runs_dir = Path("outputs/runs")
    
    completed = []
    pending = []
    
    for entry in plan:
        run_id = entry["run_id"]
        run_dir = runs_dir / run_id.replace("N/A", "N")
        status_file = run_dir / "status.json"
        
        if status_file.exists():
            try:
                status = json.load(open(status_file, encoding='utf-8'))
                if status.get("status") == "ok":
                    mtime = status_file.stat().st_mtime
                    completed.append({
                        "run_id": run_id,
                        "dataset": entry.get("dataset"),
                        "time": datetime.fromtimestamp(mtime)
                    })
            except:
                pass
        else:
            pending.append({
                "run_id": run_id,
                "dataset": entry.get("dataset")
            })
    
    total = len(plan)
    comp_count = len(completed)
    
    print("="*70)
    print(f"Experiment Progress Monitor - {datetime.now().strftime('%H:%M:%S')}")
    print("="*70)
    print(f"Total: {total} | Completed: {comp_count} ({comp_count*100/total:.1f}%) | Remaining: {len(pending)}")
    print()
    
    if completed:
        last = max(completed, key=lambda x: x["time"])
        age_minutes = (datetime.now() - last["time"]).total_seconds() / 60
        print(f"Last completed: {last['run_id']} ({last['dataset']})")
        print(f"Completed at: {last['time'].strftime('%H:%M:%S')} ({age_minutes:.1f} min ago)")
        print()
    
    if pending:
        print(f"Remaining {len(pending)} runs:")
        for p in pending[:10]:
            print(f"  {p['run_id']}: {p['dataset']}")
        if len(pending) > 10:
            print(f"  ... and {len(pending) - 10} more")
        
        # Check if any are in progress (have config but no status)
        in_progress = []
        for p in pending:
            run_dir = runs_dir / p["run_id"].replace("N/A", "N")
            if run_dir.exists() and (run_dir / "config.json").exists():
                in_progress.append(p["run_id"])
        
        if in_progress:
            print(f"\nRuns in progress (have config but no status yet):")
            for rid in in_progress[:5]:
                print(f"  {rid}")
    
    print("="*70)

if __name__ == "__main__":
    monitor()
