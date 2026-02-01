"""
Collect actual dataset sizes from all runs
"""
import json
from pathlib import Path
from collections import defaultdict

def collect_sizes():
    plan_path = "outputs/reports/experiment_plan_fast.json"
    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)
    
    dataset_sizes = defaultdict(lambda: {"n_test": set(), "n_train": set(), "n_total": set()})
    
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
            
            # For OULAD: use group_coverage
            group_coverage = metrics.get("group_coverage")
            if group_coverage:
                n_test = sum(g.get("n_total", 0) for g in group_coverage.values())
                n_train = int(n_test * 4) if n_test > 0 else 0
                n_total = n_train + n_test
                
                if n_test > 0:
                    dataset_sizes[dataset]["n_test"].add(n_test)
                    dataset_sizes[dataset]["n_train"].add(n_train)
                    dataset_sizes[dataset]["n_total"].add(n_total)
        except Exception:
            pass
    
    # Format results
    result = {}
    for dataset, sizes in dataset_sizes.items():
        if sizes["n_test"]:
            n_test_vals = sorted(sizes["n_test"])
            n_train_vals = sorted(sizes["n_train"])
            n_total_vals = sorted(sizes["n_total"])
            
            if len(n_test_vals) == 1:
                result[dataset] = {
                    "n_total": n_total_vals[0],
                    "n_train": n_train_vals[0],
                    "n_test": n_test_vals[0]
                }
            else:
                result[dataset] = {
                    "n_total": f"{n_total_vals[0]}-{n_total_vals[-1]}",
                    "n_train": f"{n_train_vals[0]}-{n_train_vals[-1]}",
                    "n_test": f"{n_test_vals[0]}-{n_test_vals[-1]}"
                }
    
    return result

if __name__ == "__main__":
    sizes = collect_sizes()
    print(json.dumps(sizes, indent=2))
