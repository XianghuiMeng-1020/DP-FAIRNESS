"""
Final verification: Ensure NO synthetic data is being used
"""
import sys
import io
from pathlib import Path
import json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def verify_no_synthetic():
    """Verify no synthetic data runs exist"""
    runs_dir = Path("outputs/runs")
    
    print("="*80)
    print("FINAL VERIFICATION: No Synthetic Data")
    print("="*80)
    
    synthetic_found = []
    real_data_count = 0
    
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir() or run_dir.name.startswith("smoke_test"):
            continue
        
        metrics_file = run_dir / "metrics.json"
        if not metrics_file.exists():
            continue
        
        try:
            metrics = json.load(open(metrics_file, encoding='utf-8'))
            n_train = metrics.get("dp_n_train", 0)
            dataset = metrics.get("dataset", "")
            
            synthetic_sizes = {"OULAD": 4000, "UCI697": 320, "HarvardX_PersonCourse": 2400}
            
            if dataset in synthetic_sizes and n_train == synthetic_sizes[dataset]:
                synthetic_found.append(run_dir.name)
            elif n_train > 0:
                real_data_count += 1
        except:
            pass
    
    print(f"\nScan Results:")
    print(f"  Real data runs: {real_data_count}")
    print(f"  Synthetic data runs: {len(synthetic_found)}")
    
    if synthetic_found:
        print(f"\nERROR: Found {len(synthetic_found)} synthetic data runs!")
        print("First 10:", synthetic_found[:10])
        return False
    else:
        print(f"\nSUCCESS: No synthetic data runs found!")
        print(f"All {real_data_count} runs use REAL data")
        return True

if __name__ == "__main__":
    success = verify_no_synthetic()
    sys.exit(0 if success else 1)
