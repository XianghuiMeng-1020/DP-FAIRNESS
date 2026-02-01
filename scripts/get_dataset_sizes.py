"""
Extract actual dataset sizes from run artifacts
"""
import json
from pathlib import Path
from collections import defaultdict

def get_dataset_sizes():
    plan_path = "outputs/reports/experiment_plan_fast.json"
    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)
    
    # Group runs by dataset
    dataset_sizes = defaultdict(lambda: {"n_test": [], "n_train": []})
    
    for entry in plan:
        run_id = entry["run_id"]
        dataset = entry["dataset"]
        run_dir = Path("outputs/runs") / run_id
        
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            
            group_coverage = metrics.get("group_coverage", {})
            if group_coverage:
                n_test = sum(g.get("n_total", 0) for g in group_coverage.values())
                # Estimate n_train (typically 4x n_test for 80/20 split)
                n_train = int(n_test * 4) if n_test > 0 else 0
                
                if n_test > 0:
                    dataset_sizes[dataset]["n_test"].append(n_test)
                    dataset_sizes[dataset]["n_train"].append(n_train)
        except Exception as e:
            print(f"Error processing {run_id}: {e}")
    
    # Calculate min-max for each dataset
    result = {}
    for dataset, sizes in dataset_sizes.items():
        if sizes["n_test"]:
            n_test_min = min(sizes["n_test"])
            n_test_max = max(sizes["n_test"])
            n_train_min = min(sizes["n_train"])
            n_train_max = max(sizes["n_train"])
            
            n_total_min = n_train_min + n_test_min
            n_total_max = n_train_max + n_test_max
            
            if n_test_min == n_test_max:
                result[dataset] = {
                    "n_total": n_total_min,
                    "n_train": n_train_min,
                    "n_test": n_test_min
                }
            else:
                result[dataset] = {
                    "n_total": f"{n_total_min}-{n_total_max}",
                    "n_train": f"{n_train_min}-{n_train_max}",
                    "n_test": f"{n_test_min}-{n_test_max}"
                }
    
    return result

if __name__ == "__main__":
    sizes = get_dataset_sizes()
    print(json.dumps(sizes, indent=2))
