"""
合理性检查脚本 - 验证真实模型训练后的结果
TASK 4: 对6个代表性运行进行合理性检查
"""
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr


def load_predictions(run_dir: Path):
    """加载predictions"""
    base_path = run_dir / "predictions_base.npy"
    released_path = run_dir / "predictions_released.npy"
    labels_path = run_dir / "test_labels.npy"
    
    predictions_base = None
    predictions_released = None
    labels = None
    
    if base_path.exists():
        predictions_base = np.load(base_path)
        if len(predictions_base.shape) > 1 and predictions_base.shape[1] == 2:
            predictions_base = predictions_base[:, 1]  # 提取正类概率
        else:
            predictions_base = predictions_base.flatten()
    
    if released_path.exists():
        predictions_released = np.load(released_path)
        if len(predictions_released.shape) > 1 and predictions_released.shape[1] == 2:
            predictions_released = predictions_released[:, 1]
        else:
            predictions_released = predictions_released.flatten()
    
    if labels_path.exists():
        labels = np.load(labels_path).flatten()
    
    return predictions_base, predictions_released, labels


def check_run(run_id: str, base_dir: str = "outputs/runs"):
    """检查单个运行"""
    run_dir = Path(base_dir) / run_id
    
    if not run_dir.exists():
        return {
            "run_id": run_id,
            "status": "missing",
            "error": "Run directory not found"
        }
    
    # 加载配置
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return {
            "run_id": run_id,
            "status": "missing_config",
            "error": "config.json not found"
        }
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # 加载predictions和labels
    predictions_base, predictions_released, labels = load_predictions(run_dir)
    
    if predictions_base is None or labels is None:
        return {
            "run_id": run_id,
            "status": "missing_artifacts",
            "error": "Missing predictions_base.npy or test_labels.npy"
        }
    
    results = {
        "run_id": run_id,
        "dataset": config.get("dataset"),
        "model": config.get("model"),
        "model_variant": config.get("model_variant"),
        "train_defense": config.get("train_defense"),
        "publish_defense": config.get("publish_defense"),
        "eps": config.get("eps"),
        "status": "ok",
    }
    
    # 1. 检查base AUC（应该是合理的，不是~0.99除非真正合理）
    base_auc = roc_auc_score(labels, predictions_base)
    results["base_auc"] = float(base_auc)
    results["base_auc_plausible"] = 0.5 <= base_auc <= 0.95  # 合理范围
    
    # 2. 检查released AUC（如果存在）
    if predictions_released is not None:
        released_auc = roc_auc_score(labels, predictions_released)
        results["released_auc"] = float(released_auc)
        results["released_auc_plausible"] = 0.5 <= released_auc <= 0.95
        
        # 3. 检查perturbation是否降低AUC（如果使用了perturbation）
        if config.get("publish_defense") == "output_perturbation":
            auc_decrease = base_auc - released_auc
            results["auc_decrease"] = float(auc_decrease)
            results["perturbation_reduces_auc"] = auc_decrease >= -0.05  # 允许小幅波动
            
            # 4. 检查rank correlation（perturbation应该降低相关性）
            rank_corr, _ = spearmanr(predictions_base, predictions_released)
            results["rank_correlation"] = float(rank_corr)
            results["rank_corr_high"] = rank_corr >= 0.8  # 应该仍然高度相关
        
        # 5. 检查coarsening是否减少唯一值（如果使用了coarsening）
        if config.get("publish_defense") == "output_coarsening":
            n_unique_base = len(np.unique(predictions_base))
            n_unique_released = len(np.unique(predictions_released))
            results["n_unique_base"] = int(n_unique_base)
            results["n_unique_released"] = int(n_unique_released)
            results["coarsening_reduces_unique"] = n_unique_released <= n_unique_base
            
            # Coarsening可能降低AUC
            auc_decrease = base_auc - released_auc
            results["auc_decrease"] = float(auc_decrease)
    
    # 6. 检查predictions分布
    results["base_pred_mean"] = float(np.mean(predictions_base))
    results["base_pred_std"] = float(np.std(predictions_base))
    results["base_pred_min"] = float(np.min(predictions_base))
    results["base_pred_max"] = float(np.max(predictions_base))
    
    # 7. 检查是否有标签泄漏的迹象（base AUC不应该异常高）
    if base_auc > 0.99:
        results["warning"] = "Base AUC > 0.99 - possible label leakage"
    
    return results


def main():
    """主函数：检查6个代表性运行"""
    # 选择6个代表性运行（不同数据集、模型、防御）
    representative_runs = [
        "fast_0000",  # OULAD, LR, none
        "fast_0005",  # OULAD, XGBoost, none
        "fast_0010",  # OULAD, MLP-small, none (假设)
        "fast_0015",  # OULAD, MLP-large, DP-SGD eps=1 (假设)
        "fast_0020",  # OULAD, LR, output_perturbation (假设)
        "fast_0025",  # OULAD, MLP-small, output_coarsening (假设)
    ]
    
    # 如果这些运行不存在，尝试从plan中获取前6个
    plan_path = Path("outputs/reports/experiment_plan_fast.json")
    if plan_path.exists():
        with open(plan_path, "r") as f:
            plan = json.load(f)
        if len(plan) >= 6:
            representative_runs = [entry["run_id"] for entry in plan[:6]]
    
    print("=" * 80)
    print("TASK 4: 合理性检查 - 真实模型训练结果")
    print("=" * 80)
    print(f"\n检查 {len(representative_runs)} 个代表性运行:\n")
    
    all_results = []
    for run_id in representative_runs:
        print(f"\n检查运行: {run_id}")
        print("-" * 80)
        result = check_run(run_id)
        all_results.append(result)
        
        if result["status"] == "ok":
            print(f"  数据集: {result['dataset']}")
            print(f"  模型: {result['model']} ({result.get('model_variant', 'N/A')})")
            print(f"  训练防御: {result['train_defense']}")
            print(f"  发布防御: {result.get('publish_defense', 'none')}")
            print(f"  Base AUC: {result['base_auc']:.4f} {'✓' if result['base_auc_plausible'] else '✗ (异常高!)'}")
            
            if "released_auc" in result:
                print(f"  Released AUC: {result['released_auc']:.4f} {'✓' if result['released_auc_plausible'] else '✗'}")
            
            if "auc_decrease" in result:
                print(f"  AUC变化: {result['auc_decrease']:.4f}")
            
            if "warning" in result:
                print(f"  警告: {result['warning']}")
        else:
            print(f"  状态: {result['status']}")
            print(f"  错误: {result.get('error', 'Unknown')}")
    
    # 汇总
    print("\n" + "=" * 80)
    print("汇总结果")
    print("=" * 80)
    
    ok_results = [r for r in all_results if r["status"] == "ok"]
    print(f"\n成功检查: {len(ok_results)}/{len(all_results)}")
    
    if ok_results:
        base_aucs = [r["base_auc"] for r in ok_results]
        print(f"\nBase AUC范围: {min(base_aucs):.4f} - {max(base_aucs):.4f}")
        print(f"Base AUC均值: {np.mean(base_aucs):.4f}")
        
        plausible_count = sum(1 for r in ok_results if r.get("base_auc_plausible", False))
        print(f"合理的Base AUC: {plausible_count}/{len(ok_results)}")
        
        # 检查是否有标签泄漏迹象
        high_auc_count = sum(1 for r in ok_results if r["base_auc"] > 0.99)
        if high_auc_count > 0:
            print(f"\n⚠️  警告: {high_auc_count} 个运行的Base AUC > 0.99 (可能的标签泄漏)")
        else:
            print("\n✓ 未发现标签泄漏迹象")
    
    # 保存结果
    output_path = Path("outputs/reports/sanity_checks_real_models.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total_checked": len(all_results),
                "ok_count": len(ok_results),
                "base_auc_range": [float(min(base_aucs)), float(max(base_aucs))] if ok_results else None,
                "base_auc_mean": float(np.mean(base_aucs)) if ok_results else None,
            },
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
