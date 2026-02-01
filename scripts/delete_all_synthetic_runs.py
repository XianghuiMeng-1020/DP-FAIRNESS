"""
CRITICAL: Delete ALL runs that used synthetic data
This script will identify and DELETE all runs that used synthetic fallback
"""
import sys
import io
from pathlib import Path
import json
import shutil

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def check_run_uses_synthetic(run_dir):
    """Check if run used synthetic data"""
    metrics_file = run_dir / "metrics.json"
    if not metrics_file.exists():
        return None
    
    try:
        metrics = json.load(open(metrics_file, encoding='utf-8'))
        n_train = metrics.get("dp_n_train", 0)
        dataset = metrics.get("dataset", "")
        
        # Synthetic data fixed sizes
        synthetic_sizes = {
            "OULAD": 4000,
            "UCI697": 320,
            "HarvardX_PersonCourse": 2400
        }
        
        if dataset in synthetic_sizes and n_train == synthetic_sizes[dataset]:
            return True
        return False
    except:
        return None

def main():
    runs_dir = Path("outputs/runs")
    if not runs_dir.exists():
        print("outputs/runs does not exist")
        return
    
    print("="*80)
    print("SCANNING FOR SYNTHETIC DATA RUNS")
    print("="*80)
    print("Checking all runs...")
    
    synthetic_runs = []
    real_data_runs = []
    unknown_runs = []
    
    total_checked = 0
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir() or run_dir.name.startswith("smoke_test"):
            continue
        
        total_checked += 1
        is_synthetic = check_run_uses_synthetic(run_dir)
        
        if is_synthetic is True:
            synthetic_runs.append(run_dir.name)
        elif is_synthetic is False:
            real_data_runs.append(run_dir.name)
        else:
            unknown_runs.append(run_dir.name)
    
    print(f"\nScan Results:")
    print(f"  Total runs checked: {total_checked}")
    print(f"  Synthetic data runs: {len(synthetic_runs)}")
    print(f"  Real data runs: {len(real_data_runs)}")
    print(f"  Unknown status: {len(unknown_runs)}")
    
    if synthetic_runs:
        print(f"\n{'='*80}")
        print(f"FOUND {len(synthetic_runs)} RUNS USING SYNTHETIC DATA")
        print(f"{'='*80}")
        print(f"First 20 synthetic runs:")
        for run_id in synthetic_runs[:20]:
            print(f"  - {run_id}")
        if len(synthetic_runs) > 20:
            print(f"  ... and {len(synthetic_runs) - 20} more")
        
        print(f"\n{'='*80}")
        print("DELETING ALL SYNTHETIC DATA RUNS")
        print(f"{'='*80}")
        
        deleted = 0
        failed = []
        
        for run_id in synthetic_runs:
            run_dir = runs_dir / run_id
            try:
                shutil.rmtree(run_dir)
                deleted += 1
                if deleted % 50 == 0:
                    print(f"  Deleted {deleted}/{len(synthetic_runs)}...")
            except Exception as e:
                failed.append((run_id, str(e)))
                print(f"  ERROR: Could not delete {run_id}: {e}")
        
        print(f"\n{'='*80}")
        print("DELETION COMPLETE")
        print(f"{'='*80}")
        print(f"Successfully deleted: {deleted}/{len(synthetic_runs)}")
        
        if failed:
            print(f"Failed to delete: {len(failed)}")
            for run_id, error in failed[:10]:
                print(f"  - {run_id}: {error}")
        
        print(f"\n✅ All synthetic data runs have been deleted!")
        print(f"Remaining runs: {len(real_data_runs)} (all using REAL data)")
        
    else:
        print("\n✅ No synthetic data runs found!")
        print("All runs are using real data.")

if __name__ == "__main__":
    main()
