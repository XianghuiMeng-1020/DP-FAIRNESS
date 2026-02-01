"""
生成contamination报告（检查wild runs）
"""
import json
from pathlib import Path
from typing import Dict, List, Set, Any

def load_plan(plan_path: str) -> List[Dict[str, Any]]:
    """加载实验计划"""
    with open(plan_path, "r", encoding="utf-8") as f:
        return json.load(f)

def scan_runs(base_dir: str = "outputs/runs") -> Set[str]:
    """扫描所有runs，返回run_id集合"""
    runs = set()
    base_path = Path(base_dir)
    
    if not base_path.exists():
        return runs
    
    for run_dir in base_path.iterdir():
        if not run_dir.is_dir():
            continue
        runs.add(run_dir.name)
    
    return runs

def generate_contamination_report(plan_path: str = "outputs/reports/experiment_plan_fast.json",
                                 base_dir: str = "outputs/runs") -> str:
    """生成contamination报告"""
    plan = load_plan(plan_path)
    expected_runs = {entry["run_id"] for entry in plan}
    actual_runs = scan_runs(base_dir)
    
    wild_runs = actual_runs - expected_runs
    missing_runs = expected_runs - actual_runs
    
    lines = []
    lines.append("# Contamination Report (Wild Runs Check)\n\n")
    lines.append("## Summary\n\n")
    lines.append(f"- **Expected runs** (from plan): {len(expected_runs)}\n")
    lines.append(f"- **Actual runs** (in outputs/runs): {len(actual_runs)}\n")
    lines.append(f"- **Wild runs** (not in plan): {len(wild_runs)}\n")
    lines.append(f"- **Missing runs** (in plan but not executed): {len(missing_runs)}\n")
    lines.append("\n")
    
    lines.append("## Status\n\n")
    if len(wild_runs) == 0:
        lines.append("**✓ PASS**: No wild runs detected. All runs are from the plan.\n\n")
    else:
        lines.append(f"**✗ FAIL**: Found {len(wild_runs)} wild runs (runs not in plan).\n\n")
    
    if wild_runs:
        lines.append("## Wild Runs List\n\n")
        lines.append("The following runs are present in outputs/runs but not in the plan:\n\n")
        for run_id in sorted(wild_runs)[:50]:  # 最多显示50个
            lines.append(f"- `{run_id}`\n")
        if len(wild_runs) > 50:
            lines.append(f"\n... and {len(wild_runs) - 50} more wild runs\n")
        lines.append("\n")
    
    if missing_runs:
        lines.append("## Missing Runs List\n\n")
        lines.append("The following runs are in the plan but not yet executed:\n\n")
        for run_id in sorted(missing_runs)[:50]:  # 最多显示50个
            lines.append(f"- `{run_id}`\n")
        if len(missing_runs) > 50:
            lines.append(f"\n... and {len(missing_runs) - 50} more missing runs\n")
        lines.append("\n")
    
    lines.append("## Decision Rule\n\n")
    lines.append("**Requirement**: Wild runs = 0 (all runs must be from the plan)\n\n")
    lines.append("This ensures reproducibility and prevents contamination from ad-hoc experiments.\n\n")
    
    return "".join(lines)

def main():
    plan_path = "outputs/reports/experiment_plan_fast.json"
    base_dir = "outputs/runs"
    
    report = generate_contamination_report(plan_path, base_dir)
    
    output_path = Path("outputs/reports/contamination_report.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"Generated contamination report")
    print(f"Saved to: {output_path}")
    
    # 检查是否有wild runs
    plan = load_plan(plan_path)
    expected_runs = {entry["run_id"] for entry in plan}
    actual_runs = scan_runs(base_dir)
    wild_runs = actual_runs - expected_runs
    
    if wild_runs:
        print(f"WARNING: Found {len(wild_runs)} wild runs")
        exit(1)
    else:
        print("✓ No wild runs detected")

if __name__ == "__main__":
    main()
