"""
Check if existing runs use synthetic data, clean if needed
"""
import sys
import io
from pathlib import Path
import json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def check_run_uses_synthetic(run_dir):
    """检查运行是否使用合成数据"""
    metrics_file = run_dir / "metrics.json"
    if not metrics_file.exists():
        return None
    
    try:
        metrics = json.load(open(metrics_file, encoding='utf-8'))
        n_train = metrics.get("dp_n_train", 0)
        dataset = metrics.get("dataset", "")
        
        # 合成数据的固定大小
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
        print("outputs/runs 不存在")
        return
    
    print("Checking existing runs...")
    synthetic_runs = []
    real_data_runs = []
    unknown_runs = []
    
    total = 0
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir() or run_dir.name.startswith("smoke_test"):
            continue
        total += 1
        if total > 100:  # Sample first 100
            break
        
        is_synthetic = check_run_uses_synthetic(run_dir)
        if is_synthetic is True:
            synthetic_runs.append(run_dir.name)
        elif is_synthetic is False:
            real_data_runs.append(run_dir.name)
        else:
            unknown_runs.append(run_dir.name)
    
    print(f"\nCheck results (sampled {total} runs):")
    print(f"  Synthetic data runs: {len(synthetic_runs)}")
    print(f"  Real data runs: {len(real_data_runs)}")
    print(f"  Unknown status: {len(unknown_runs)}")
    
    if synthetic_runs:
        print(f"\nFound {len(synthetic_runs)} runs using synthetic data")
        print(f"First 10: {synthetic_runs[:10]}")
        print("\nWARNING: Old synthetic runs detected!")
        print("Recommendation: Delete old runs and start fresh")
    else:
        print("\nOK: No synthetic data runs detected in sample")

if __name__ == "__main__":
    main()
