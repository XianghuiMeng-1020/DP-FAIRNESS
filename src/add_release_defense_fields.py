"""
为现有的 metrics.json 添加 release defense 字段
"""
import json
from pathlib import Path
from typing import Dict, Any

def load_plan(plan_path: str) -> list:
    """加载实验计划"""
    with open(plan_path, "r", encoding="utf-8") as f:
        return json.load(f)

def add_release_defense_fields(metrics: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """为 metrics 添加 release defense 字段"""
    updated = metrics.copy()
    
    # Release defense 参数（必须字段）
    release_visibility = config.get("visibility", "full")
    attack_input_visibility = "full"  # 攻击端通常看 full（除非特殊设计）
    topk_k = None
    rounding_step = config.get("coarsening_step")
    noise_std = config.get("noise_scale")
    
    # 如果字段不存在，添加它们
    if "release_visibility" not in updated:
        updated["release_visibility"] = release_visibility
    if "attack_input_visibility" not in updated:
        updated["attack_input_visibility"] = attack_input_visibility
    if "topk_k" not in updated:
        updated["topk_k"] = topk_k
    if "rounding_step" not in updated:
        updated["rounding_step"] = rounding_step
    if "noise_std" not in updated:
        updated["noise_std"] = noise_std
    
    return updated

def main():
    """为所有 runs 的 metrics.json 添加 release defense 字段"""
    plan_path = "outputs/reports/experiment_plan_fast.json"
    plan = load_plan(plan_path)
    base_dir = Path("outputs/runs")
    
    print(f"Adding release defense fields to metrics for {len(plan)} runs...")
    
    updated_count = 0
    for entry in plan:
        run_id = entry["run_id"]
        
        metrics_path = base_dir / run_id / "metrics.json"
        config_path = base_dir / run_id / "config.json"
        
        if not metrics_path.exists() or not config_path.exists():
            continue
        
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            
            # 检查是否需要更新
            if "release_visibility" not in metrics:
                updated_metrics = add_release_defense_fields(metrics, config)
                
                with open(metrics_path, "w", encoding="utf-8") as f:
                    json.dump(updated_metrics, f, indent=2, ensure_ascii=False)
                
                updated_count += 1
                if updated_count % 50 == 0:
                    print(f"  Updated {updated_count} runs...")
        
        except Exception as e:
            print(f"  Error updating {run_id}: {e}")
    
    print(f"\nDone! Updated {updated_count} runs.")

if __name__ == "__main__":
    main()
