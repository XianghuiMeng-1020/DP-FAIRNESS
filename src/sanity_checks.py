"""
Paper Sanity Checks Suite
检测所有可疑的退化数字、缺失负控制、占位符等
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import numpy as np
import statistics

def load_plan(plan_path: str) -> List[Dict[str, Any]]:
    """加载实验计划"""
    with open(plan_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_metrics(run_id: str, base_dir: str = "outputs/runs") -> Optional[Dict[str, Any]]:
    """加载单个run的metrics"""
    metrics_path = Path(base_dir) / run_id / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        with open(metrics_path, "r") as f:
            return json.load(f)
    except:
        return None

def load_excluded_runs() -> set:
    """Load excluded runs from paper/excluded_runs.json"""
    excluded_path = Path("paper/excluded_runs.json")
    if not excluded_path.exists():
        return set()
    try:
        with open(excluded_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return set(data.get("excluded_runs", []))
    except:
        return set()

def aggregate_metrics(plan: List[Dict[str, Any]], base_dir: str = "outputs/runs", 
                      core_only: bool = False) -> Dict[Tuple, List[Dict[str, Any]]]:
    """聚合所有metrics"""
    excluded_runs = load_excluded_runs()
    aggregated = defaultdict(list)
    
    for entry in plan:
        if core_only and not entry.get("is_core", False):
            continue
        
        run_id = entry["run_id"]
        # Skip excluded runs
        if run_id in excluded_runs:
            continue
            
        metrics = load_metrics(run_id, base_dir)
        
        if metrics is None:
            continue
        
        key = (
            entry["dataset"],
            entry["model"],
            entry.get("model_variant"),
            entry["train_defense"],
            entry.get("publish_defense") or "none",
            entry.get("eps"),
            entry["visibility"],
            entry["fairness_attribute"],
        )
        
        combined = {**entry, **metrics}
        aggregated[key].append(combined)
    
    return aggregated

def check_degenerate_invariance(aggregated: Dict[Tuple, List[Dict[str, Any]]], 
                                metric_name: str, threshold: float = 0.7, 
                                decimals: int = 4) -> Dict[str, Any]:
    """
    检查退化不变性：对于应该在不同条件下变化的指标，检测是否>70%的行共享相同的值（到4位小数）
    """
    all_values = []
    for key, runs in aggregated.items():
        for run in runs:
            value = run.get(metric_name)
            if value is not None and isinstance(value, (int, float)):
                all_values.append(round(value, decimals))
    
    if len(all_values) == 0:
        return {
            "pass": True,
            "metric": metric_name,
            "reason": "No values found",
        }
    
    # 统计每个值的出现次数
    value_counts = defaultdict(int)
    for val in all_values:
        value_counts[val] += 1
    
    # 找到最常见的值及其频率
    if len(value_counts) == 0:
        return {"pass": True, "metric": metric_name, "reason": "No valid values"}
    
    max_count = max(value_counts.values())
    max_freq = max_count / len(all_values)
    most_common_value = max(value_counts.items(), key=lambda x: x[1])[0]
    
    pass_check = max_freq <= threshold
    
    return {
        "pass": pass_check,
        "metric": metric_name,
        "total_values": len(all_values),
        "unique_values": len(value_counts),
        "most_common_value": most_common_value,
        "most_common_frequency": max_freq,
        "threshold": threshold,
        "message": f"{metric_name}: {max_freq*100:.1f}% of values are {most_common_value} (threshold: {threshold*100}%)" if not pass_check else f"{metric_name}: OK (max frequency {max_freq*100:.1f}%)",
    }

def check_seed_variance(aggregated: Dict[Tuple, List[Dict[str, Any]]], 
                        metric_name: str, min_seeds: int = 5, 
                        epsilon: float = 1e-6) -> Dict[str, Any]:
    """
    检查seed方差：对于seeds≥5的设置，至少一个关键指标必须有非零std（>epsilon）
    """
    failures = []
    
    for key, runs in aggregated.items():
        if len(runs) < min_seeds:
            continue
        
        values = [r.get(metric_name) for r in runs if r.get(metric_name) is not None]
        if len(values) < min_seeds:
            continue
        
        std_val = statistics.stdev(values) if len(values) > 1 else 0.0
        
        if std_val <= epsilon:
            dataset, model, variant, train_def, pub_def, eps, visibility, fairness = key
            failures.append({
                "setting": f"{dataset}/{model}/{variant}/{train_def}/{pub_def}/eps={eps}",
                "metric": metric_name,
                "std": std_val,
                "values": values[:5],  # 前5个值
                "seed_count": len(values),
            })
    
    return {
        "pass": len(failures) == 0,
        "metric": metric_name,
        "failures": failures,
        "total_settings_checked": len([k for k, v in aggregated.items() if len(v) >= min_seeds]),
        "failed_settings": len(failures),
    }

def check_threat_model_sanity(aggregated: Dict[Tuple, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    威胁模型合理性：在Table 10.5（或等价表）中，切换release_visibility或attack_input_visibility
    必须改变相关攻击指标（至少一个配对比较显示非零效应）
    """
    # 按 (dataset, model, variant, train_def, eps) 分组，比较不同 visibility
    visibility_groups = defaultdict(lambda: defaultdict(list))
    
    for key, runs in aggregated.items():
        dataset, model, variant, train_def, pub_def, eps, visibility, fairness = key
        if pub_def != "none" and pub_def is not None:
            continue  # 只比较无 publish_defense 的情况
        if fairness != "gender" and fairness != "NA":
            continue
        
        group_key = (dataset, model, variant, train_def, eps)
        visibility_groups[group_key][visibility].extend(runs)
    
    failures = []
    
    for group_key, visibility_dict in visibility_groups.items():
        if "full" not in visibility_dict:
            continue
        
        full_runs = visibility_dict["full"]
        full_mia_aucs = [r.get("mia_auc") for r in full_runs if r.get("mia_auc") is not None]
        if len(full_mia_aucs) == 0:
            continue
        
        full_mean = statistics.mean(full_mia_aucs)
        
        for vis in visibility_dict.keys():
            if vis == "full":
                continue
            
            vis_runs = visibility_dict[vis]
            vis_mia_aucs = [r.get("mia_auc") for r in vis_runs if r.get("mia_auc") is not None]
            if len(vis_mia_aucs) == 0:
                continue
            
            vis_mean = statistics.mean(vis_mia_aucs)
            change = abs(vis_mean - full_mean)
            
            if change < 1e-6:  # 变化太小
                dataset, model, variant, train_def, eps = group_key
                failures.append({
                    "setting": f"{dataset}/{model}/{variant}/{train_def}/eps={eps}",
                    "full_visibility_auc": full_mean,
                    f"{vis}_visibility_auc": vis_mean,
                    "change": change,
                })
    
    return {
        "pass": len(failures) == 0,
        "failures": failures,
        "total_comparisons": sum(len(v) - 1 for v in visibility_groups.values() if "full" in v),
        "failed_comparisons": len(failures),
    }

def check_fairness_metric_definitions(aggregated: Dict[Tuple, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    公平性指标定义合理性：
    1. worst-group gaps应该是max_g - min_g
    2. gaps应该在[0,1]范围内
    3. 检测如果"gap"列意外存储了原始rates（例如值~0.98到处都是）
    """
    failures = []
    
    gap_metrics = ["worst_group_tpr_gap", "worst_group_fpr_gap", "worst_group_fnr_gap"]
    
    for metric_name in gap_metrics:
        all_values = []
        for key, runs in aggregated.items():
            for run in runs:
                value = run.get(metric_name)
                if value is not None and isinstance(value, (int, float)):
                    all_values.append(value)
        
        if len(all_values) == 0:
            continue
        
        # 检查范围
        out_of_range = [v for v in all_values if v < 0 or v > 1]
        if len(out_of_range) > 0:
            failures.append({
                "metric": metric_name,
                "issue": "out_of_range",
                "count": len(out_of_range),
                "examples": out_of_range[:5],
            })
        
        # 检查是否看起来像原始rate而不是gap（值接近1.0）
        suspicious_high = [v for v in all_values if v > 0.9]
        if len(suspicious_high) > len(all_values) * 0.5:  # 超过50%的值>0.9
            failures.append({
                "metric": metric_name,
                "issue": "suspicious_high_values",
                "count": len(suspicious_high),
                "total": len(all_values),
                "percentage": len(suspicious_high) / len(all_values) * 100,
                "mean": statistics.mean(all_values),
                "message": f"{metric_name}: {len(suspicious_high)}/{len(all_values)} values > 0.9, mean={statistics.mean(all_values):.5f} (suspicious: might be raw TPR instead of gap)",
            })
    
    return {
        "pass": len(failures) == 0,
        "failures": failures,
    }

def check_range_sanity(aggregated: Dict[Tuple, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    范围合理性检查：
    - AUC应该在[0,1]
    - ECE应该在[0,1]
    - gaps应该在[0,1]
    - 没有NaN除非明确标记为"not evaluated"
    """
    failures = []
    
    range_checks = {
        "test_auc": (0, 1),
        "mia_auc": (0, 1),
        "ece": (0, 1),
        "worst_group_tpr_gap": (0, 1),
        "worst_group_fpr_gap": (0, 1),
        "worst_group_fnr_gap": (0, 1),
        "group_ece": (0, 1),
    }
    
    for metric_name, (min_val, max_val) in range_checks.items():
        all_values = []
        for key, runs in aggregated.items():
            for run in runs:
                value = run.get(metric_name)
                if value is not None:
                    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                        failures.append({
                            "metric": metric_name,
                            "issue": "nan_or_inf",
                            "value": value,
                        })
                    elif isinstance(value, (int, float)):
                        all_values.append(value)
                        if value < min_val or value > max_val:
                            failures.append({
                                "metric": metric_name,
                                "issue": "out_of_range",
                                "value": value,
                                "expected_range": (min_val, max_val),
                            })
    
    return {
        "pass": len(failures) == 0,
        "failures": failures,
    }

def check_negative_controls(plan: List[Dict[str, Any]], base_dir: str = "outputs/runs") -> Dict[str, Any]:
    """
    检查负控制是否存在：
    - Random Labels: utility AUC ≈ 0.5
    - Random Groups (OULAD): fairness gaps ≈ 0
    """
    # 查找负控制runs（通过run_id或特殊标记）
    negative_control_runs = []
    
    for entry in plan:
        run_id = entry["run_id"]
        # 检查是否有负控制标记（可能在config中）
        if "negative_control" in entry.get("tags", []) or "random_labels" in run_id.lower() or "random_groups" in run_id.lower():
            negative_control_runs.append(entry)
    
    if len(negative_control_runs) == 0:
        return {
            "pass": False,
            "issue": "missing",
            "message": "No negative control runs found in plan",
        }
    
    # 检查负控制的metrics
    results = []
    for entry in negative_control_runs:
        run_id = entry["run_id"]
        metrics = load_metrics(run_id, base_dir)
        
        if metrics is None:
            results.append({
                "run_id": run_id,
                "status": "missing_metrics",
            })
            continue
        
        # Random Labels检查：test_auc应该≈0.5
        if "random_labels" in run_id.lower():
            test_auc = metrics.get("test_auc")
            if test_auc is None:
                results.append({"run_id": run_id, "status": "missing_test_auc"})
            elif abs(test_auc - 0.5) > 0.1:  # 允许0.1的容差
                results.append({
                    "run_id": run_id,
                    "status": "unexpected_value",
                    "test_auc": test_auc,
                    "expected": "≈0.5",
                })
            else:
                results.append({"run_id": run_id, "status": "ok", "test_auc": test_auc})
        
        # Random Groups检查：fairness gaps应该≈0
        if "random_groups" in run_id.lower():
            tpr_gap = metrics.get("worst_group_tpr_gap")
            if tpr_gap is None:
                results.append({"run_id": run_id, "status": "missing_tpr_gap"})
            elif abs(tpr_gap) > 0.05:  # 允许0.05的容差
                results.append({
                    "run_id": run_id,
                    "status": "unexpected_value",
                    "tpr_gap": tpr_gap,
                    "expected": "≈0",
                })
            else:
                results.append({"run_id": run_id, "status": "ok", "tpr_gap": tpr_gap})
    
    all_ok = all(r.get("status") == "ok" for r in results)
    
    return {
        "pass": all_ok,
        "negative_control_runs": len(negative_control_runs),
        "results": results,
    }

def run_all_sanity_checks(plan_path: str = "outputs/reports/experiment_plan_fast.json",
                          base_dir: str = "outputs/runs") -> Dict[str, Any]:
    """运行所有合理性检查"""
    plan = load_plan(plan_path)
    aggregated = aggregate_metrics(plan, base_dir, core_only=True)
    
    results = {}
    
    # 1. 退化不变性检查
    print("Checking degenerate invariance...")
    key_metrics = ["test_auc", "mia_auc", "worst_group_tpr_gap", "ece", "group_ece"]
    results["degenerate_invariance"] = {}
    for metric in key_metrics:
        results["degenerate_invariance"][metric] = check_degenerate_invariance(aggregated, metric)
    
    # 2. Seed方差检查
    print("Checking seed variance...")
    results["seed_variance"] = {}
    for metric in key_metrics:
        results["seed_variance"][metric] = check_seed_variance(aggregated, metric)
    
    # 3. 威胁模型合理性
    print("Checking threat model sanity...")
    results["threat_model"] = check_threat_model_sanity(aggregated)
    
    # 4. 公平性指标定义
    print("Checking fairness metric definitions...")
    results["fairness_definitions"] = check_fairness_metric_definitions(aggregated)
    
    # 5. 范围合理性
    print("Checking range sanity...")
    results["range_sanity"] = check_range_sanity(aggregated)
    
    # 6. 负控制检查
    print("Checking negative controls...")
    results["negative_controls"] = check_negative_controls(plan, base_dir)
    
    # 总体通过/失败
    all_pass = (
        all(r["pass"] for r in results["degenerate_invariance"].values()) and
        all(r["pass"] for r in results["seed_variance"].values()) and
        results["threat_model"]["pass"] and
        results["fairness_definitions"]["pass"] and
        results["range_sanity"]["pass"] and
        results["negative_controls"]["pass"]
    )
    
    results["overall_pass"] = all_pass
    
    return results

def generate_sanity_report(results: Dict[str, Any], output_path: str = "paper/sanity_report.md") -> str:
    """生成合理性检查报告"""
    lines = []
    lines.append("# Paper Sanity Report\n\n")
    lines.append("This report summarizes all sanity checks performed on the paper metrics.\n\n")
    lines.append(f"**Overall Status**: {'✅ PASS' if results['overall_pass'] else '❌ FAIL'}\n\n")
    lines.append("---\n\n")
    
    # 1. 退化不变性
    lines.append("## 1. Degenerate Invariance Check\n\n")
    lines.append("**Purpose**: Detect metrics that should vary across conditions but show near-constant values.\n\n")
    for metric, result in results["degenerate_invariance"].items():
        status = "✅ PASS" if result["pass"] else "❌ FAIL"
        lines.append(f"### {metric}\n\n")
        lines.append(f"**Status**: {status}\n\n")
        if not result["pass"]:
            lines.append(f"- {result['message']}\n")
        else:
            lines.append(f"- OK: Max frequency {result['most_common_frequency']*100:.1f}%\n")
        lines.append(f"- Total values: {result['total_values']}\n")
        lines.append(f"- Unique values: {result['unique_values']}\n\n")
    
    # 2. Seed方差
    lines.append("## 2. Seed Variance Check\n\n")
    lines.append("**Purpose**: For settings with seeds≥5, at least one key metric must have non-zero std.\n\n")
    for metric, result in results["seed_variance"].items():
        status = "✅ PASS" if result["pass"] else "❌ FAIL"
        lines.append(f"### {metric}\n\n")
        lines.append(f"**Status**: {status}\n\n")
        if not result["pass"]:
            lines.append(f"- Failed settings: {result['failed_settings']}\n")
            for failure in result["failures"][:5]:  # 显示前5个失败
                lines.append(f"  - {failure['setting']}: std={failure['std']:.6f}, values={failure['values']}\n")
        else:
            lines.append(f"- OK: All {result['total_settings_checked']} settings have variance\n\n")
    
    # 3. 威胁模型
    lines.append("## 3. Threat Model Sanity Check\n\n")
    lines.append("**Purpose**: Switching visibility must change attack metrics.\n\n")
    status = "✅ PASS" if results["threat_model"]["pass"] else "❌ FAIL"
    lines.append(f"**Status**: {status}\n\n")
    if not results["threat_model"]["pass"]:
        lines.append(f"- Failed comparisons: {results['threat_model']['failed_comparisons']}\n")
        for failure in results["threat_model"]["failures"][:5]:
            lines.append(f"  - {failure['setting']}: change={failure['change']:.6f}\n")
    else:
        lines.append(f"- OK: All {results['threat_model']['total_comparisons']} comparisons show effect\n\n")
    
    # 4. 公平性定义
    lines.append("## 4. Fairness Metric Definitions Check\n\n")
    lines.append("**Purpose**: Validate gaps are computed as max-min and within [0,1].\n\n")
    status = "✅ PASS" if results["fairness_definitions"]["pass"] else "❌ FAIL"
    lines.append(f"**Status**: {status}\n\n")
    if not results["fairness_definitions"]["pass"]:
        for failure in results["fairness_definitions"]["failures"]:
            lines.append(f"- {failure['metric']}: {failure['issue']}\n")
            if "message" in failure:
                lines.append(f"  - {failure['message']}\n")
    else:
        lines.append("- OK: All fairness gaps are correctly defined\n\n")
    
    # 5. 范围合理性
    lines.append("## 5. Range Sanity Check\n\n")
    lines.append("**Purpose**: Check AUC in [0,1], ECE in [0,1], gaps in [0,1], no NaN.\n\n")
    status = "✅ PASS" if results["range_sanity"]["pass"] else "❌ FAIL"
    lines.append(f"**Status**: {status}\n\n")
    if not results["range_sanity"]["pass"]:
        for failure in results["range_sanity"]["failures"][:10]:
            lines.append(f"- {failure['metric']}: {failure['issue']}\n")
            if 'value' in failure:
                lines.append(f"  - Value: {failure['value']}\n")
    else:
        lines.append("- OK: All metrics are within valid ranges\n\n")
    
    # 6. 负控制
    lines.append("## 6. Negative Controls Check\n\n")
    lines.append("**Purpose**: Verify negative controls exist and have expected values.\n\n")
    status = "✅ PASS" if results["negative_controls"]["pass"] else "❌ FAIL"
    lines.append(f"**Status**: {status}\n\n")
    if not results["negative_controls"]["pass"]:
        if "message" in results["negative_controls"]:
            lines.append(f"- {results['negative_controls']['message']}\n")
        for result in results["negative_controls"].get("results", []):
            if result.get("status") != "ok":
                lines.append(f"- {result['run_id']}: {result['status']}\n")
    else:
        lines.append(f"- OK: {results['negative_controls']['negative_control_runs']} negative control runs found and validated\n\n")
    
    return "".join(lines)

def main():
    plan_path = "outputs/reports/experiment_plan_fast.json"
    base_dir = "outputs/runs"
    
    print("Running sanity checks...")
    results = run_all_sanity_checks(plan_path, base_dir)
    
    # 生成报告
    report_content = generate_sanity_report(results)
    
    output_path = Path("paper/sanity_report.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print(f"\nSanity report saved to: {output_path}")
    print(f"Overall status: {'PASS' if results['overall_pass'] else 'FAIL'}")
    
    # 如果失败，退出非零
    if not results["overall_pass"]:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
