"""
修复threat model：更新所有runs的attack_input_visibility字段
规则：
- 如果publish_defense == "output_coarsening": attack_input_visibility = "full" (stronger-than-release)
- 否则: attack_input_visibility = release_visibility (same-as-release)
"""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

def load_plan(plan_path: str):
    """加载实验计划"""
    with open(plan_path, "r", encoding="utf-8") as f:
        return json.load(f)

def fix_threat_model_for_run(run_dir: Path, entry) -> bool:
    """修复单个run的threat model"""
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return False
    
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    
    release_visibility = metrics.get("release_visibility") or entry.get("visibility", "full")
    publish_def = entry.get("publish_defense") or metrics.get("publish_defense")
    
    # 确定attack_input_visibility
    if publish_def == "output_coarsening":
        # Coarsening只影响release端，攻击端仍看full（stronger-than-release）
        attack_input_visibility = "full"
    else:
        # 默认：攻击端看到与release端相同的信息（same-as-release）
        attack_input_visibility = release_visibility
    
    # 更新metrics
    updated = False
    if metrics.get("attack_input_visibility") != attack_input_visibility:
        metrics["attack_input_visibility"] = attack_input_visibility
        updated = True
    
    if metrics.get("release_visibility") != release_visibility:
        metrics["release_visibility"] = release_visibility
        updated = True
    
    if updated:
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        return True
    
    return False

def main():
    plan_path = "outputs/reports/experiment_plan_fast.json"
    base_dir = "outputs/runs"
    
    plan = load_plan(plan_path)
    fixed_count = 0
    
    for entry in plan:
        run_id = entry["run_id"]
        run_dir = Path(base_dir) / run_id
        
        if not run_dir.exists():
            continue
        
        if fix_threat_model_for_run(run_dir, entry):
            fixed_count += 1
            print(f"Fixed {run_id}")
    
    print(f"\nFixed {fixed_count} runs")

if __name__ == "__main__":
    main()
