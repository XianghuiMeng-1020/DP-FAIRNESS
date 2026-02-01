"""
Diagnose why experiment is stuck
"""
import sys
import io
from pathlib import Path
import json
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.run_all import load_plan, get_run_dir

def diagnose():
    """Diagnose stuck runs"""
    plan = load_plan("outputs/reports/experiment_plan_fast.json")
    runs_dir = Path("outputs/runs")
    
    print("="*80)
    print("Diagnosing Stuck Experiment")
    print("="*80)
    
    # Find pending runs
    pending = []
    for entry in plan:
        run_id = entry["run_id"]
        run_dir = get_run_dir(run_id)
        status_file = run_dir / "status.json"
        
        if not status_file.exists():
            pending.append({
                "run_id": run_id,
                "dataset": entry.get("dataset"),
                "has_dir": run_dir.exists(),
                "has_config": (run_dir / "config.json").exists(),
                "has_partial": any(run_dir.glob("*.npy")) if run_dir.exists() else False
            })
    
    print(f"\nPending runs: {len(pending)}")
    print(f"\nFirst 10 pending runs:")
    for p in pending[:10]:
        print(f"  {p['run_id']} ({p['dataset']}) - dir exists: {p['has_dir']}, has config: {p['has_config']}, has partial files: {p['has_partial']}")
    
    # Check if any run is currently being processed (recent modification)
    print(f"\nChecking for runs in progress...")
    import time
    current_time = time.time()
    in_progress = []
    
    for p in pending[:20]:
        run_dir = get_run_dir(p["run_id"])
        if run_dir.exists():
            # Check modification time
            try:
                mtime = max([f.stat().st_mtime for f in run_dir.iterdir() if f.is_file()])
                age_minutes = (current_time - mtime) / 60
                if age_minutes < 5:  # Modified in last 5 minutes
                    in_progress.append((p["run_id"], age_minutes))
            except:
                pass
    
    if in_progress:
        print(f"Runs modified in last 5 minutes:")
        for run_id, age in in_progress:
            print(f"  {run_id} ({age:.1f} min ago)")
    else:
        print("No runs modified in last 5 minutes - experiment appears stuck")
    
    # Try to run first pending run to see error
    if pending:
        first_pending = pending[0]
        print(f"\n{'='*80}")
        print(f"Testing first pending run: {first_pending['run_id']}")
        print(f"{'='*80}")
        
        # Find entry
        entry = next((e for e in plan if e["run_id"] == first_pending["run_id"]), None)
        if entry:
            print(f"Config: {json.dumps(entry, indent=2, default=str)}")
            print(f"\nAttempting to run...")
            try:
                from src.run_all import run_experiment
                result = run_experiment(entry, base_dir="outputs/runs")
                print(f"Result: {result.get('status')}")
                if result.get("status") != "ok":
                    print(f"Error: {result.get('error', 'Unknown')}")
            except Exception as e:
                print(f"Exception: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    diagnose()
