"""
获取6个代表性run的prediction/attack文件hash作为证据
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from recompute_from_artifacts import recompute_metrics_from_artifacts, compute_file_hash

def get_representative_runs(plan_path: str = "outputs/reports/experiment_plan_fast.json") -> list:
    """获取6个代表性runs"""
    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)
    
    # 选择core runs，按setting分组，选择第一个run
    core_runs = [entry for entry in plan if entry.get("is_core", False)]
    
    seen_settings = set()
    representative_runs = []
    
    for entry in core_runs:
        setting_key = (
            entry["dataset"],
            entry["model"],
            entry.get("model_variant"),
            entry["train_defense"],
            entry.get("publish_defense") or "none",
            entry.get("eps"),
            entry["visibility"],
            entry["fairness_attribute"],
        )
        
        if setting_key not in seen_settings:
            seen_settings.add(setting_key)
            representative_runs.append(entry["run_id"])
    
    # 限制为6个
    return representative_runs[:6]

def main():
    base_dir = "outputs/runs"
    representative_runs = get_representative_runs()
    
    print("=" * 60)
    print("Representative Runs File Hashes")
    print("=" * 60)
    print()
    
    results = []
    
    for run_id in representative_runs:
        run_dir = Path(base_dir) / run_id
        
        if not run_dir.exists():
            print(f"{run_id}: Run directory not found")
            continue
        
        # 获取recompute结果（包含file hashes）
        result = recompute_metrics_from_artifacts(run_dir, tolerance=1e-6)
        
        file_hashes = result.get("file_hashes", {})
        
        print(f"Run ID: {run_id}")
        print(f"  Config: {run_dir / 'config.json'}")
        
        if file_hashes:
            print("  Artifact Files:")
            for artifact_type, hash_info in file_hashes.items():
                file_name = hash_info.get("file", "unknown")
                file_hash = hash_info.get("hash", "N/A")
                file_path = run_dir / file_name
                print(f"    {artifact_type}: {file_name}")
                print(f"      Hash (SHA256): {file_hash}")
                print(f"      Path: {file_path}")
        else:
            print("  No artifact files found")
        
        print()
        
        results.append({
            "run_id": run_id,
            "file_hashes": file_hashes,
            "recompute_status": result.get("status"),
            "all_pass": result.get("all_pass", False),
        })
    
    # 保存结果
    output_path = Path("outputs/reports/representative_runs_hashes.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()
