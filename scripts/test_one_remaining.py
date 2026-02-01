"""
Test one remaining run to see why it's stuck
"""
import sys
import io
from pathlib import Path
import json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.run_all import load_plan, get_run_dir

def main():
    plan = load_plan("outputs/reports/experiment_plan_fast.json")
    runs_dir = Path("outputs/runs")
    
    # Find first remaining run
    remaining = None
    for entry in plan:
        run_id = entry["run_id"]
        run_dir = get_run_dir(run_id)
        status_file = run_dir / "status.json"
        
        if not status_file.exists():
            remaining = entry
            break
    
    if not remaining:
        print("No remaining runs!")
        return
    
    run_id = remaining["run_id"]
    run_dir = get_run_dir(run_id)
    
    print(f"Testing remaining run: {run_id}")
    print(f"Dataset: {remaining.get('dataset')}")
    print(f"Model: {remaining.get('model')}")
    print(f"Run dir exists: {run_dir.exists()}")
    
    if run_dir.exists():
        files = list(run_dir.iterdir())
        print(f"Files in dir: {[f.name for f in files[:10]]}")
    
    print(f"\nAttempting to run {run_id}...")
    print("(This may take a while for HarvardX dataset)")
    
    try:
        from src.run_all import run_experiment
        result = run_experiment(remaining)
        print(f"\nResult: {result.get('status')}")
        if result.get('status') == 'ok':
            print("SUCCESS!")
        else:
            print(f"Error: {result.get('error', 'Unknown')}")
    except Exception as e:
        print(f"\nException: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
