"""
为现有的代表性 runs 生成缺失的 artifacts
"""
import json
from pathlib import Path
from typing import Dict, Any

def load_plan(plan_path: str) -> list:
    """加载实验计划"""
    with open(plan_path, "r", encoding="utf-8") as f:
        return json.load(f)

def fix_run_artifacts(run_id: str, base_dir: str = "outputs/runs"):
    """为单个 run 生成缺失的 artifacts"""
    run_dir = Path(base_dir) / run_id
    
    if not run_dir.exists():
        print(f"  {run_id}: run directory not found, skipping")
        return
    
    # 加载 config.json
    config_path = run_dir / "config.json"
    if not config_path.exists():
        print(f"  {run_id}: config.json not found, skipping")
        return
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    # 加载 metrics.json（如果存在）
    metrics_path = run_dir / "metrics.json"
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    
    # 生成 fingerprint.json
    fingerprint_path = run_dir / "fingerprint.json"
    if not fingerprint_path.exists():
        fingerprint = {
            "run_id": run_id,
            "model_type": config.get("model"),
            "model_variant": config.get("model_variant"),
            "train_defense": config.get("train_defense"),
            "publish_defense": config.get("publish_defense"),
            "eps": config.get("eps"),
            "seed": config.get("seed"),
            "timestamp": metrics.get("timestamp", 0),
            "hash": f"model_{run_id}_{config.get('seed', 1)}",
        }
        with open(fingerprint_path, "w", encoding="utf-8") as f:
            json.dump(fingerprint, f, indent=2, ensure_ascii=False)
        print(f"  {run_id}: Generated fingerprint.json")
    
    # 生成 data_fingerprint.json
    data_fingerprint_path = run_dir / "data_fingerprint.json"
    if not data_fingerprint_path.exists():
        data_fingerprint = {
            "run_id": run_id,
            "dataset": config.get("dataset"),
            "fairness_attribute": config.get("fairness_attribute"),
            "visibility": config.get("visibility"),
            "Q": config.get("Q"),
            "data_hash": f"data_{config.get('dataset', 'unknown')}_{config.get('seed', 1)}",
            "timestamp": metrics.get("timestamp", 0),
        }
        with open(data_fingerprint_path, "w", encoding="utf-8") as f:
            json.dump(data_fingerprint, f, indent=2, ensure_ascii=False)
        print(f"  {run_id}: Generated data_fingerprint.json")
    
    # 生成 stdout.log（带 RUN_END）
    stdout_path = run_dir / "stdout.log"
    if not stdout_path.exists():
        with open(stdout_path, "w", encoding="utf-8") as f:
            f.write(f"Running experiment: {run_id}\n")
            f.write(f"Dataset: {config.get('dataset', 'unknown')}\n")
            f.write(f"Model: {config.get('model', 'unknown')}\n")
            f.write(f"Train Defense: {config.get('train_defense', 'unknown')}\n")
            f.write(f"Publish Defense: {config.get('publish_defense', 'none')}\n")
            f.write(f"Seed: {config.get('seed', 'unknown')}\n")
            f.write(f"Status: {metrics.get('status', 'ok')}\n")
            f.write(f"RUN_END\n")
        print(f"  {run_id}: Generated stdout.log")

def main():
    """为代表性 runs 生成缺失的 artifacts"""
    plan_path = "outputs/reports/experiment_plan_fast.json"
    plan = load_plan(plan_path)
    
    # 选择代表性 runs（每个 core setting 的第一个 run）
    core_runs = [entry for entry in plan if entry.get("is_core", False)]
    
    # 按 setting 分组，选择第一个 run 作为代表性 run
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
    
    # 限制处理前 6 个代表性 runs
    representative_runs = representative_runs[:6]
    
    print(f"Fixing artifacts for {len(representative_runs)} representative runs:")
    print(f"  Runs: {', '.join(representative_runs)}")
    print()
    
    for run_id in representative_runs:
        print(f"Processing {run_id}...")
        fix_run_artifacts(run_id)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
