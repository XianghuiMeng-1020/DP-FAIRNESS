"""
Verify that completed runs used real data
"""
import json
from pathlib import Path

def verify():
    plan = json.load(open("outputs/reports/experiment_plan_fast.json", encoding='utf-8'))
    runs_dir = Path("outputs/runs")
    
    synthetic_sizes = {"OULAD": 4000, "UCI697": 320, "HarvardX_PersonCourse": 2400}
    
    print("Verifying real data usage in completed runs...")
    print("="*70)
    
    synthetic_found = []
    real_data_count = 0
    
    for entry in plan:
        run_id = entry["run_id"]
        run_dir = runs_dir / run_id.replace("N/A", "N")
        metrics_file = run_dir / "metrics.json"
        
        if not metrics_file.exists():
            continue
        
        try:
            metrics = json.load(open(metrics_file, encoding='utf-8'))
            n_train = metrics.get("dp_n_train", 0)
            dataset = metrics.get("dataset", "")
            
            if dataset in synthetic_sizes and n_train == synthetic_sizes[dataset]:
                synthetic_found.append(run_id)
            elif n_train > 0:
                real_data_count += 1
        except:
            pass
    
    print(f"Total runs checked: {len(plan)}")
    print(f"Real data runs: {real_data_count}")
    print(f"Synthetic data runs: {len(synthetic_found)}")
    
    if synthetic_found:
        print(f"\nWARNING: Found {len(synthetic_found)} runs using synthetic data!")
        print("First 10:", synthetic_found[:10])
        return False
    else:
        print(f"\nSUCCESS: All runs use real data!")
        return True

if __name__ == "__main__":
    success = verify()
