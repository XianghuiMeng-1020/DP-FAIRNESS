"""
Find remaining runs and test one
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
    
    remaining = []
    for entry in plan:
        run_id = entry["run_id"]
        run_dir = get_run_dir(run_id)
        status_file = run_dir / "status.json"
        
        if not status_file.exists():
            remaining.append(entry)
    
    print(f"Remaining runs: {len(remaining)}")
    print("\nFirst 5 remaining runs:")
    for e in remaining[:5]:
        print(f"  {e['run_id']}: {e.get('dataset')} | {e.get('model')} | seed={e.get('seed')}")
    
    if remaining:
        print(f"\nTesting first remaining run: {remaining[0]['run_id']}")
        print(f"Config: {json.dumps(remaining[0], indent=2, default=str)}")
        
        # Try to run it
        try:
            from src.run_all import run_experiment
            print("\nAttempting to run...")
            result = run_experiment(remaining[0])
            print(f"Result status: {result.get('status')}")
            if result.get('status') != 'ok':
                print(f"Error: {result.get('error', 'Unknown')}")
        except Exception as e:
            print(f"Exception: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
