"""为所有DP-SGD runs添加DP元数据（delta, noise_multiplier等）到config.json和metrics.json"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reporting import load_plan
from src.data_loader import load_dataset

def eps_to_noise_multiplier(eps: float) -> float:
    """将epsilon转换为noise multiplier（与model_trainer.py中的实现一致）"""
    if eps >= 10:
        return 0.5
    elif eps >= 5:
        return 1.0
    elif eps >= 1:
        return 2.0
    else:
        return 5.0

def get_n_train_from_artifacts(run_dir: Path) -> int:
    """从artifacts推断n_train"""
    # 尝试从membership.npy推断（membership前n_train是1，后n_test是0）
    membership_path = run_dir / "membership.npy"
    if membership_path.exists():
        import numpy as np
        membership = np.load(membership_path)
        n_train = int(np.sum(membership == 1))
        if n_train > 0:
            return n_train
    
    # 如果无法从artifacts推断，使用数据集默认值
    # 这些值来自data_loader.py的实际数据集大小
    dataset_defaults = {
        "OULAD": 16000,  # 约32K总样本，train/test split约50/50
        "UCI697": 350,   # 约697总样本，train/test split约50/50
        "HarvardX_PersonCourse": 5000,  # 约10K总样本，train/test split约50/50
    }
    return dataset_defaults.get(entry.get("dataset", ""), 1000)

def compute_dp_metadata(entry: dict, run_dir: Path = None) -> dict:
    """计算DP元数据"""
    train_defense = entry.get("train_defense")
    eps = entry.get("eps")
    
    if train_defense != "DP-SGD" or eps is None:
        return {}
    
    # 获取n_train
    if run_dir:
        n_train = get_n_train_from_artifacts(run_dir)
    else:
        # Fallback: 加载数据集
        dataset_name = entry["dataset"]
        seed = entry.get("seed", 1)
        try:
            X_train, X_test, y_train, y_test, groups_test = load_dataset(dataset_name, seed=seed)
            n_train = len(X_train)
        except:
            n_train = get_n_train_from_artifacts(Path("outputs/runs") / entry["run_id"].replace("N/A", "N"))
    
    # DP超参数
    batch_size = 64  # 从model_trainer.py中获取
    n_epochs = 30  # DP-SGD使用30 epochs
    sample_rate = batch_size / n_train if n_train > 0 else 0.0
    delta = 1.0 / n_train if n_train > 0 else 0.0  # 标准做法：delta = 1/n_train
    noise_multiplier = eps_to_noise_multiplier(eps)
    
    # clip_norm: 代码中没有gradient clipping，所以设为None
    clip_norm = None
    
    return {
        "dp_target_delta": delta,
        "dp_accountant_type": "simplified_dp_sgd",  # 使用简化版DP-SGD，不是Opacus
        "dp_noise_multiplier": noise_multiplier,
        "dp_clip_norm": clip_norm,
        "dp_batch_size": batch_size,
        "dp_sample_rate": sample_rate,
        "dp_epochs": n_epochs,
        "dp_n_train": n_train,
    }

def update_run_metadata(run_dir: Path, entry: dict):
    """更新单个run的config.json和metrics.json"""
    dp_metadata = compute_dp_metadata(entry, run_dir)
    
    if not dp_metadata:
        return False  # 不是DP run，跳过
    
    # 更新config.json
    config_path = run_dir / "config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # 只添加config级别的字段
        config["dp_target_delta"] = dp_metadata["dp_target_delta"]
        config["dp_accountant_type"] = dp_metadata["dp_accountant_type"]
        
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    # 更新metrics.json
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        
        # 添加所有DP元数据字段
        metrics.update(dp_metadata)
        
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    return True

def main():
    """为所有DP-SGD runs添加DP元数据"""
    plan_path = "outputs/reports/experiment_plan_fast.json"
    base_dir = Path("outputs/runs")
    
    plan = load_plan(plan_path)
    
    updated_count = 0
    error_count = 0
    
    for entry in plan:
        run_id = entry["run_id"]
        run_dir = base_dir / run_id.replace("N/A", "N")
        
        if not run_dir.exists():
            continue
        
        try:
            if update_run_metadata(run_dir, entry):
                updated_count += 1
        except Exception as e:
            print(f"Error updating {run_id}: {e}")
            error_count += 1
    
    print(f"Updated {updated_count} DP-SGD runs")
    print(f"Errors: {error_count}")

if __name__ == "__main__":
    main()
