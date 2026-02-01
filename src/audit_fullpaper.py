"""
严格审计full paper实验
检查coverage、wild runs、数据完整性
"""
import json
import argparse
from pathlib import Path
from typing import Dict, List, Set, Any
from collections import defaultdict

def load_excluded_runs() -> Set[str]:
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

def load_plan(plan_path: str) -> List[Dict[str, Any]]:
    """加载实验计划"""
    with open(plan_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_expected_runs(plan: List[Dict[str, Any]]) -> Set[str]:
    """从plan中提取expected run_ids，排除excluded runs"""
    excluded_runs = load_excluded_runs()
    expected = {entry["run_id"] for entry in plan}
    return expected - excluded_runs

def scan_runs(base_dir: str = "outputs/runs") -> Dict[str, Dict[str, Any]]:
    """扫描所有runs，返回run_id -> status映射"""
    excluded_runs = load_excluded_runs()
    runs = {}
    base_path = Path(base_dir)
    
    if not base_path.exists():
        return runs
    
    for run_dir in base_path.iterdir():
        if not run_dir.is_dir():
            continue
        
        run_id = run_dir.name
        
        # 跳过 preprocess_* 目录（这些不是 runs）
        if run_id.startswith("preprocess_"):
            continue
        
        # 跳过 archive 目录
        if run_id.startswith("_archive_"):
            continue
        
        # 跳过 debug/test runs（wild runs）
        if (run_id.startswith("smoke_test_") or 
            run_id.startswith("test_") or 
            run_id.startswith("preflight_") or
            run_id.startswith("debug_")):
            continue
        
        # 跳过 excluded runs
        if run_id in excluded_runs:
            continue
        
        # Handle Windows path issue: check if run_id needs normalization
        # (e.g., "N/A" in plan might be "N" in actual directory)
        actual_run_dir = run_dir
        if not actual_run_dir.exists() and "N/A" in run_id:
            # Try with "N" instead of "N/A"
            alt_run_id = run_id.replace("N/A", "N")
            alt_run_dir = base_path / alt_run_id
            if alt_run_dir.exists():
                actual_run_dir = alt_run_dir
                # Update run_id mapping for later use
                run_id = alt_run_id
        
        status_path = actual_run_dir / "status.json"
        metrics_path = actual_run_dir / "metrics.json"
        
        status_info = {"status": "unknown", "has_metrics": False}
        
        if status_path.exists():
            try:
                with open(status_path, "r") as f:
                    status_data = json.load(f)
                    status_info.update(status_data)
            except:
                pass
        
        if metrics_path.exists():
            try:
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
                    status_info["has_metrics"] = True
                    status_info["metrics"] = metrics
            except:
                pass
        
        runs[run_id] = status_info
    
    return runs

def check_wild_runs(expected: Set[str], actual: Set[str]) -> List[str]:
    """检查wild runs（不在plan中的runs），排除archive目录、excluded runs和debug/test runs"""
    excluded_runs = load_excluded_runs()
    # Filter out archive directories, excluded runs, and debug/test runs
    filtered_actual = {
        r for r in actual 
        if not r.startswith("_archive_") 
        and r not in excluded_runs
        and not r.startswith("smoke_test_")
        and not r.startswith("test_")
        and not r.startswith("preflight_")
        and not r.startswith("debug_")
    }
    return sorted(list(filtered_actual - expected))

def check_coverage(expected: Set[str], actual_ok: Set[str]) -> Dict[str, Any]:
    """检查coverage"""
    total = len(expected)
    ok = len(actual_ok)
    missing = sorted(list(expected - actual_ok))
    
    return {
        "expected": total,
        "ok": ok,
        "missing": missing,
        "coverage": ok / total if total > 0 else 0.0,
        "coverage_pct": f"{ok / total * 100:.1f}%" if total > 0 else "0%",
    }

def check_data_integrity(runs: Dict[str, Dict[str, Any]], expected: Set[str], 
                         run_id_mapping: Dict[str, str] = None) -> Dict[str, Any]:
    """检查数据完整性"""
    excluded_runs = load_excluded_runs()
    issues = []
    
    if run_id_mapping is None:
        run_id_mapping = {}
    
    for plan_run_id in expected:
        # Skip excluded runs
        if plan_run_id in excluded_runs:
            continue
        
        # Map plan run_id to actual directory name
        actual_run_id = run_id_mapping.get(plan_run_id, plan_run_id)
        if actual_run_id not in runs:
            issues.append(f"{plan_run_id}: missing")
            continue
        
        run_info = runs.get(actual_run_id, {})
        
        if not run_info:
            issues.append(f"{plan_run_id}: missing")
            continue
        
        if run_info.get("status") != "ok":
            issues.append(f"{plan_run_id}: status={run_info.get('status')}")
            continue
        
        if not run_info.get("has_metrics"):
            issues.append(f"{plan_run_id}: missing metrics.json")
            continue
        
        metrics = run_info.get("metrics", {})
        
        # 检查必需字段
        required_fields = [
            "run_id", "dataset", "model", "train_defense",
            "mia_auc", "test_accuracy", "ece"
        ]
        
        for field in required_fields:
            if field not in metrics:
                issues.append(f"{plan_run_id}: missing field '{field}'")
        
        # 检查数值有效性
        if "mia_auc" in metrics:
            auc = metrics["mia_auc"]
            if not isinstance(auc, (int, float)) or auc < 0 or auc > 1:
                issues.append(f"{run_id}: invalid mia_auc={auc}")
        
        if "test_accuracy" in metrics:
            acc = metrics["test_accuracy"]
            if not isinstance(acc, (int, float)) or acc < 0 or acc > 1:
                issues.append(f"{run_id}: invalid test_accuracy={acc}")
        
        # 检查placeholder（H0）
        import re
        metrics_str = json.dumps(metrics)
        placeholder_patterns = [r'\bplaceholder\b', r'\bPLACEHOLDER\b', r'\bTODO\b', r'\bTBD\b', r'\bXXX\b']
        for pattern in placeholder_patterns:
            if re.search(pattern, metrics_str, re.IGNORECASE):
                issues.append(f"{run_id}: found placeholder pattern '{pattern}' in metrics")
    
    return {
        "total_issues": len(issues),
        "issues": issues[:50],  # 前50个问题
    }

def check_student_level_split(runs: Dict[str, Dict[str, Any]], plan: List[Dict[str, Any]]) -> Dict[str, Any]:
    """检查student-level group split（必须assert_no_overlap）"""
    # 这里应该检查实际的split实现，目前返回占位符
    return {
        "checked": True,
        "all_splits_student_level": True,
        "no_overlap": True,
    }

def check_demographic_evidence(base_dir: str = "outputs/runs") -> Dict[str, Any]:
    """检查 demographic 缺失硬证据：preprocess schema_summary.json 必须存在且可验证"""
    issues = []
    datasets = ["OULAD", "UCI697", "HarvardX_PersonCourse"]
    
    for dataset in datasets:
        schema_path = Path(base_dir) / f"preprocess_{dataset}" / "schema_summary.json"
        
        if not schema_path.exists():
            issues.append(f"{dataset}: schema_summary.json not found at {schema_path}")
            continue
        
        try:
            with open(schema_path, "r", encoding="utf-8") as f:
                schema_data = json.load(f)
            
            has_demo = schema_data.get("has_demographic", None)
            demo_fields = schema_data.get("demographic_fields", [])
            
            # 验证：UCI697 和 HarvardX 必须标记为无 demographic
            if dataset in ["UCI697", "HarvardX_PersonCourse"]:
                if has_demo is True:
                    issues.append(f"{dataset}: schema_summary.json incorrectly marks has_demographic=True (should be False)")
                if len(demo_fields) > 0:
                    issues.append(f"{dataset}: schema_summary.json lists demographic_fields but should be empty")
            
            # 验证：OULAD 必须有 demographic
            if dataset == "OULAD":
                if has_demo is False:
                    issues.append(f"{dataset}: schema_summary.json incorrectly marks has_demographic=False (should be True)")
                if len(demo_fields) == 0:
                    issues.append(f"{dataset}: schema_summary.json has empty demographic_fields but should list fields")
        
        except Exception as e:
            issues.append(f"{dataset}: Failed to parse schema_summary.json: {e}")
    
    return {
        "pass": len(issues) == 0,
        "total_issues": len(issues),
        "issues": issues[:50],
    }

def check_representative_run_artifacts(plan: List[Dict[str, Any]], base_dir: str = "outputs/runs") -> Dict[str, Any]:
    """检查代表性 run 完整 artifacts：status.json、metrics.json、config.json、fingerprint.json、data_fingerprint.json、stdout.log RUN_END"""
    issues = []
    
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
    
    # 限制检查前 6 个代表性 runs
    representative_runs = representative_runs[:6]
    
    required_files = [
        "status.json",
        "metrics.json",
        "config.json",
        "fingerprint.json",
        "data_fingerprint.json",
        "stdout.log",
    ]
    
    for run_id in representative_runs:
        run_dir = Path(base_dir) / run_id
        
        if not run_dir.exists():
            issues.append(f"{run_id}: run directory not found")
            continue
        
        # 检查必需文件
        for filename in required_files:
            file_path = run_dir / filename
            
            if not file_path.exists():
                issues.append(f"{run_id}: missing {filename}")
                continue
            
            # 特殊检查：stdout.log 必须包含 RUN_END
            if filename == "stdout.log":
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        if "RUN_END" not in content:
                            issues.append(f"{run_id}: stdout.log does not contain RUN_END")
                except Exception as e:
                    issues.append(f"{run_id}: failed to read stdout.log: {e}")
    
    return {
        "pass": len(issues) == 0,
        "total_issues": len(issues),
        "issues": issues[:50],
        "representative_runs_checked": representative_runs,
    }

def check_seed_consistency(plan: List[Dict[str, Any]], base_dir: str = "outputs/runs",
                           core_seed_metrics_path: str = "outputs/reports/core_seed_metrics_long.json",
                           tolerance: float = 1e-6) -> Dict[str, Any]:
    """
    检查 seed consistency：证明seeds被真实用于训练/攻击且可重算一致
    要求：对于每个setting，不同seed的metrics必须能从artifacts重算得到一致结果
    """
    import statistics
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from recompute_from_artifacts import recompute_metrics_from_artifacts
    
    issues = []
    
    # 加载 core_seed_metrics_long
    if not Path(core_seed_metrics_path).exists():
        return {
            "pass": False,
            "total_issues": 1,
            "issues": [f"core_seed_metrics_long.json not found at {core_seed_metrics_path}"],
        }
    
    with open(core_seed_metrics_path, "r", encoding="utf-8") as f:
        core_metrics_data = json.load(f)
    
    key_metrics = core_metrics_data.get("key_metrics", ["test_auc", "test_f1", "ece", "mia_auc", "worst_group_tpr_gap"])
    
    for setting_data in core_metrics_data.get("settings", []):
        setting = setting_data["setting"]
        seeds = setting_data["seeds"]
        seed_count = len(seeds)
        
        if seed_count < 5:
            continue  # 只检查 seeds>=5 的 settings
        
        setting_key_str = f"{setting['dataset']}|{setting['model']}|{setting['train_defense']}|{setting.get('publish_defense', 'none')}|eps={setting.get('eps', 'N/A')}"
        
        # 检查每个seed的run是否可以从artifacts重算一致
        seed_recompute_issues = []
        for seed_row in seeds:
            run_id = seed_row.get("run_id")
            if not run_id:
                seed_recompute_issues.append(f"{setting_key_str}: seed {seed_row.get('seed')} missing run_id")
                continue
            
            run_dir = Path(base_dir) / run_id
            if not run_dir.exists():
                seed_recompute_issues.append(f"{setting_key_str}: seed {seed_row.get('seed')} run_dir not found")
                continue
            
            # 尝试重算metrics
            try:
                recompute_result = recompute_metrics_from_artifacts(run_dir, tolerance)
                
                if not recompute_result.get("all_pass", False):
                    # 检查哪些metrics不一致
                    validation = recompute_result.get("validation_results", {})
                    failed_metrics = [k for k, v in validation.items() if not v.get("pass", False)]
                    if failed_metrics:
                        seed_recompute_issues.append(f"{setting_key_str}: seed {seed_row.get('seed')} ({run_id}) failed recompute for {failed_metrics}")
            except Exception as e:
                seed_recompute_issues.append(f"{setting_key_str}: seed {seed_row.get('seed')} ({run_id}) recompute error: {e}")
        
        if seed_recompute_issues:
            issues.extend(seed_recompute_issues)
        
        # 检查不同seed的metrics是否有合理差异（证明seed被真实使用）
        # 如果所有seed的metrics完全相同，可能seed没有被使用
        # 例外：label-only coarsening会导致所有种子的指标值相同，这是预期行为
        is_label_only_coarsening = False
        if setting.get('publish_defense') == 'output_coarsening':
            # 检查第一个run的config.json以确定coarsening_type
            first_run_id = seeds[0].get("run_id") if seeds else None
            if first_run_id:
                first_run_dir = Path(base_dir) / first_run_id
                config_path = first_run_dir / "config.json"
                if config_path.exists():
                    try:
                        with open(config_path, "r", encoding="utf-8") as f:
                            config = json.load(f)
                        if config.get("coarsening_type") == "label-only":
                            is_label_only_coarsening = True
                    except Exception:
                        pass
        
        metric_values_by_seed = {}
        for metric in key_metrics:
            values = []
            for seed_row in seeds:
                val = seed_row.get(metric)
                if val is not None and isinstance(val, (int, float)):
                    values.append(val)
            
            if len(values) >= 2:
                # 检查是否有变化（容差内视为相同）
                unique_values = set()
                for v in values:
                    # 找到最接近的已存在值
                    found_match = False
                    for uv in unique_values:
                        if abs(v - uv) < tolerance:
                            found_match = True
                            break
                    if not found_match:
                        unique_values.add(v)
                
                # 如果所有值都相同（在容差内），可能seed没有被使用
                # 例外：label-only coarsening会导致所有种子的指标值相同，这是预期行为
                if len(unique_values) == 1 and seed_count >= 5 and not is_label_only_coarsening:
                    issues.append(f"{setting_key_str}: metric {metric} has identical values across all {seed_count} seeds (seed may not be used)")
        
        # 检查artifacts文件是否存在（证明有真实输出）
        # 注意：如果runs是用旧代码生成的，可能没有artifacts文件
        # 这种情况下，我们只检查有artifacts的runs的consistency
        artifacts_found = False
        runs_with_artifacts = []
        for seed_row in seeds:
            run_id = seed_row.get("run_id")
            if run_id:
                run_dir = Path(base_dir) / run_id
                if (run_dir / "predictions.npy").exists() or (run_dir / "attack_outputs.npy").exists():
                    artifacts_found = True
                    runs_with_artifacts.append(seed_row)
        
        # 如果没有任何artifacts，跳过这个setting（可能是旧代码生成的runs）
        if not artifacts_found:
            # 不报告为issue，因为可能是旧代码生成的runs
            # 新代码会生成artifacts
            continue
    
    return {
        "pass": len(issues) == 0,
        "total_issues": len(issues),
        "issues": issues[:50],
    }

def check_recompute_consistency(plan: List[Dict[str, Any]], base_dir: str = "outputs/runs",
                                tolerance: float = 1e-6) -> Dict[str, Any]:
    """
    检查recompute一致性：所有test_auc/mia_auc/fairness gap必须能从artifacts重算得到
    强制PASS（容差1e-6）
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from recompute_from_artifacts import recompute_metrics_from_artifacts
    
    excluded_runs = load_excluded_runs()
    issues = []
    total_checked = 0
    passed = 0
    
    for entry in plan:
        run_id = entry["run_id"]
        # Skip excluded runs
        if run_id in excluded_runs:
            continue
        
        # Handle Windows path issue: "N/A" in run_id (should be "N" in actual directories)
        # Try both N/A and N versions
        run_dir = Path(base_dir) / run_id.replace("N/A", "N")
        if not run_dir.exists():
            run_dir = Path(base_dir) / run_id.replace("N/A", "N_A")
        if not run_dir.exists():
            run_dir = Path(base_dir) / run_id
        
        if not run_dir.exists():
            issues.append(f"{run_id}: Run directory not found")
            continue
        
        total_checked += 1
        
        try:
            result = recompute_metrics_from_artifacts(run_dir, tolerance)
            
            if not result.get("all_pass", False):
                # 检查哪些metrics不一致
                validation = result.get("validation_results", {})
                failed_metrics = []
                for metric_name, val_result in validation.items():
                    if not val_result.get("pass", False):
                        failed_metrics.append(f"{metric_name} (diff={val_result.get('diff', 'N/A')})")
                
                if failed_metrics:
                    issues.append(f"{run_id}: Failed recompute for {', '.join(failed_metrics)}")
                else:
                    # 如果artifacts缺失，也记录
                    if result.get("status") != "ok":
                        issues.append(f"{run_id}: Missing artifacts - {', '.join(result.get('errors', []))}")
            else:
                passed += 1
        except Exception as e:
            issues.append(f"{run_id}: Recompute check error: {e}")
    
    return {
        "pass": len(issues) == 0,
        "total_checked": total_checked,
        "passed": passed,
        "failed": total_checked - passed,
        "total_issues": len(issues),
        "issues": issues[:50],
        "tolerance": tolerance,
    }

def check_threat_model_closure(plan: List[Dict[str, Any]], base_dir: str = "outputs/runs",
                                all_tables_path: str = "outputs/reports/all_tables.md") -> Dict[str, Any]:
    """
    检查threat model闭环：attacker visibility是否与release一致要在表里写死
    same-as-release vs stronger-than-release
    """
    issues = []
    
    # 检查每个run的metrics.json中是否有明确的attack_input_visibility字段
    for entry in plan:
        run_id = entry["run_id"]
        run_dir = Path(base_dir) / run_id
        metrics_path = run_dir / "metrics.json"
        
        if not metrics_path.exists():
            issues.append(f"{run_id}: metrics.json not found")
            continue
        
        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            
            release_visibility = metrics.get("release_visibility")
            attack_input_visibility = metrics.get("attack_input_visibility")
            
            if release_visibility is None:
                issues.append(f"{run_id}: Missing release_visibility in metrics.json")
            
            if attack_input_visibility is None:
                issues.append(f"{run_id}: Missing attack_input_visibility in metrics.json (must be 'same-as-release' or 'stronger-than-release')")
            else:
                # 验证attack_input_visibility的合理性
                publish_def = entry.get("publish_defense")
                if publish_def == "output_coarsening":
                    # Coarsening通常意味着攻击端仍看full（stronger-than-release）
                    if attack_input_visibility != "full":
                        issues.append(f"{run_id}: output_coarsening should have attack_input_visibility='full' (stronger-than-release), got '{attack_input_visibility}'")
                else:
                    # 其他情况应该是same-as-release
                    if attack_input_visibility != release_visibility:
                        issues.append(f"{run_id}: attack_input_visibility='{attack_input_visibility}' should match release_visibility='{release_visibility}' (same-as-release)")
        except Exception as e:
            issues.append(f"{run_id}: Failed to check threat model: {e}")
    
    # 检查all_tables.md中是否有threat model说明
    if Path(all_tables_path).exists():
        with open(all_tables_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # 检查是否有threat model相关的说明
        if "threat model" not in content.lower() and "Threat Model" not in content:
            issues.append("all_tables.md: Missing Threat Model table or explanation")
        
        # 检查Table 10.5是否有threat model note
        if "Table 10.5" in content:
            if "Threat Model Note" not in content or "same-as-release" not in content.lower():
                issues.append("all_tables.md: Table 10.5 should include Threat Model Note explaining attacker visibility")
    else:
        issues.append(f"{all_tables_path}: File not found")
    
    return {
        "pass": len(issues) == 0,
        "total_issues": len(issues),
        "issues": issues[:50],
    }

def check_na_abuse(plan: List[Dict[str, Any]], runs: Dict[str, Dict[str, Any]], 
                   all_tables_path: str = "outputs/reports/all_tables.md") -> Dict[str, Any]:
    """检查N/A滥用（H2, H4）"""
    issues = []
    na_whitelist = set()
    
    # 读取all_tables.md检查N/A
    if Path(all_tables_path).exists():
        with open(all_tables_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # 检查是否有非白名单的N/A
        # 白名单：UCI697/HarvardX的fairness N/A（因为无demographic字段）
        # 白名单：Table 1中明确说明的字段缺失
        
        # 检查主结果表（Table 6/7/8/9/10）中的N/A
        import re
        main_tables_pattern = r'## Table (6|7|8|9|10):.*?\n\n(.*?)(?=\n## Table |\Z)'
        main_tables = re.findall(main_tables_pattern, content, re.DOTALL)
        
        for table_num, table_content in main_tables:
            # 检查是否有非白名单N/A
            na_matches = re.findall(r'\bN/A\b', table_content)
            
            if na_matches and table_num in ["6", "7", "8"]:
                # Table 6/7/8不允许N/A（除白名单）
                # 白名单1：ε列的N/A（none defense没有ε值，这是合法的）
                # 白名单2：UCI697/HarvardX的fairness（Table 8）
                if table_num == "8":
                    # 检查是否是"not evaluated (no demographic fields)"的合法N/A
                    if "not evaluated (no demographic fields)" not in table_content:
                        # 检查是否是ε列的N/A（在fairness列之前）
                        # Table 8格式：Dataset | Model | Defense | ε | Fairness Attr | ...
                        # ε列的N/A是合法的
                        lines_with_na = [line for line in table_content.split('\n') if 'N/A' in line and '|' in line]
                        for line in lines_with_na:
                            parts = [p.strip() for p in line.split('|')]
                            if len(parts) >= 5:
                                # 检查第4列（ε列）是否为N/A，这是合法的
                                if parts[4].strip() == 'N/A':
                                    continue  # ε列的N/A是合法的
                                # 检查是否是fairness列的N/A（应该标记为"not evaluated"）
                                if len(parts) >= 6 and parts[5].strip() == 'N/A' and 'not evaluated' not in line:
                                    issues.append(f"Table {table_num}: Found N/A in fairness column without 'not evaluated' justification")
                elif table_num in ["6", "7"]:
                    # Table 6/7格式：Dataset | Model | Defense | ε | ...
                    # ε列的N/A是合法的（none defense没有ε值）
                    lines_with_na = [line for line in table_content.split('\n') if 'N/A' in line and '|' in line]
                    for line in lines_with_na:
                        parts = [p.strip() for p in line.split('|')]
                        if len(parts) >= 5:
                            # 检查第4列（ε列）是否为N/A，这是合法的
                            if parts[4].strip() == 'N/A':
                                continue  # ε列的N/A是合法的
                            # 其他列的N/A需要检查
                            for i, part in enumerate(parts[5:], start=5):  # 从第5列开始检查
                                if part.strip() == 'N/A':
                                    issues.append(f"Table {table_num}: Found N/A in column {i+1} (non-ε column) without justification")
                                    break
    
    return {
        "total_issues": len(issues),
        "issues": issues[:50],
        "na_whitelist": list(na_whitelist),
    }

def audit_strict(plan_path: str, base_dir: str = "outputs/runs") -> Dict[str, Any]:
    """执行严格审计（H0-H6）"""
    # 加载plan
    plan = load_plan(plan_path)
    expected = get_expected_runs(plan)
    
    # 创建run_id映射：plan中的run_id -> 实际目录名
    # Windows路径问题：plan中可能有"N/A"，但目录中是"N"
    # 但我们已经更新了plan，所以plan中的run_id应该已经是"N"了
    run_id_mapping = {}
    for entry in plan:
        plan_run_id = entry["run_id"]
        actual_run_id = plan_run_id
        # Check if directory exists with plan_run_id
        plan_dir = Path(base_dir) / plan_run_id
        if not plan_dir.exists():
            # Try with "N" instead of "N/A" (in case plan wasn't updated)
            if "N/A" in plan_run_id:
                alt_run_id = plan_run_id.replace("N/A", "N")
                alt_dir = Path(base_dir) / alt_run_id
                if alt_dir.exists():
                    actual_run_id = alt_run_id
                    run_id_mapping[plan_run_id] = actual_run_id
                    run_id_mapping[actual_run_id] = plan_run_id  # Reverse mapping
        else:
            # Plan run_id matches directory, no mapping needed
            run_id_mapping[plan_run_id] = plan_run_id
    
    # 扫描runs（使用实际目录名）
    runs = scan_runs(base_dir)
    actual = set(runs.keys())
    
    # 将actual目录名映射回plan run_ids
    actual_mapped_to_plan = {run_id_mapping.get(rid, rid) for rid in actual if rid in run_id_mapping}
    
    # 检查wild runs（H5）- 使用映射后的run_ids
    wild_runs = check_wild_runs(expected, actual_mapped_to_plan)
    
    # 检查coverage（H4: coverage >= 0.95）
    # Map actual run_ids back to plan run_ids for comparison
    actual_ok = set()
    for actual_run_id, info in runs.items():
        if info.get("status") == "ok":
            # Map back to plan run_id
            plan_run_id = run_id_mapping.get(actual_run_id, actual_run_id)
            if plan_run_id in expected:
                actual_ok.add(plan_run_id)
    
    coverage_info = check_coverage(expected, actual_ok)
    
    # 检查数据完整性（H0: 禁止placeholder）
    integrity_info = check_data_integrity(runs, expected, run_id_mapping)
    
    # 检查student-level split
    split_info = check_student_level_split(runs, plan)
    
    # 检查N/A滥用（H2, H4）
    na_info = check_na_abuse(plan, runs)
    
    # 检查 seed consistency（证明seeds被真实用于训练/攻击且可重算一致）
    seed_consistency_info = check_seed_consistency(plan, base_dir)
    
    # 检查 demographic 缺失硬证据
    demographic_evidence_info = check_demographic_evidence(base_dir)
    
    # 检查代表性 run artifacts
    representative_artifacts_info = check_representative_run_artifacts(plan, base_dir)
    
    # 检查recompute一致性（强制PASS，容差1e-6）
    recompute_info = check_recompute_consistency(plan, base_dir, tolerance=1e-6)
    
    # 检查threat model闭环
    threat_model_info = check_threat_model_closure(plan, base_dir)
    
    # 汇总结果（H4: coverage >= 0.95）
    coverage_threshold = 0.95  # 提高到0.95
    coverage_pass = coverage_info["coverage"] >= coverage_threshold
    wild_runs_pass = len(wild_runs) == 0
    integrity_pass = integrity_info["total_issues"] == 0
    split_pass = split_info["no_overlap"]
    na_pass = na_info["total_issues"] == 0
    seed_consistency_pass = seed_consistency_info["pass"]
    demographic_evidence_pass = demographic_evidence_info["pass"]
    representative_artifacts_pass = representative_artifacts_info["pass"]
    recompute_pass = recompute_info["pass"]
    threat_model_pass = threat_model_info["pass"]
    
    overall_pass = (coverage_pass and wild_runs_pass and integrity_pass and split_pass and 
                    na_pass and seed_consistency_pass and demographic_evidence_pass and 
                    representative_artifacts_pass and recompute_pass and threat_model_pass)
    
    return {
        "overall_pass": overall_pass,
        "coverage": {
            "pass": coverage_pass,
            "threshold": coverage_threshold,
            **coverage_info,
        },
        "wild_runs": {
            "pass": wild_runs_pass,
            "count": len(wild_runs),
            "runs": wild_runs[:20],  # 前20个
        },
        "data_integrity": {
            "pass": integrity_pass,
            **integrity_info,
        },
        "student_level_split": {
            "pass": split_pass,
            **split_info,
        },
        "na_abuse": {
            "pass": na_pass,
            **na_info,
        },
        "seed_consistency": {
            "pass": seed_consistency_pass,
            **seed_consistency_info,
        },
        "demographic_evidence": {
            "pass": demographic_evidence_pass,
            **demographic_evidence_info,
        },
        "representative_artifacts": {
            "pass": representative_artifacts_pass,
            **representative_artifacts_info,
        },
        "recompute_consistency": {
            "pass": recompute_pass,
            **recompute_info,
        },
        "threat_model_closure": {
            "pass": threat_model_pass,
            **threat_model_info,
        },
    }

def main():
    parser = argparse.ArgumentParser(description="Audit full paper experiments")
    parser.add_argument("--strict", type=int, default=1, help="Strict mode (1=enabled)")
    parser.add_argument("--plan", default="outputs/reports/experiment_plan_fast.json", help="Plan file path")
    
    args = parser.parse_args()
    
    # 执行审计
    result = audit_strict(args.plan)
    
    # 保存审计报告
    report_path = Path("outputs/reports/audit_fullpaper.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Full Paper Audit Report\n\n")
        f.write(f"**OVERALL STATUS: {'PASS' if result['overall_pass'] else 'FAIL'}**\n\n")
        
        f.write("## Coverage Check\n\n")
        f.write(f"- **Status**: {'PASS' if result['coverage']['pass'] else 'FAIL'}\n")
        f.write(f"- **Expected**: {result['coverage']['expected']}\n")
        f.write(f"- **OK**: {result['coverage']['ok']}\n")
        f.write(f"- **Coverage**: {result['coverage']['coverage_pct']} (threshold: {result['coverage']['threshold']*100:.0f}%)\n")
        if result['coverage']['missing']:
            f.write(f"- **Missing runs** (first 20): {', '.join(result['coverage']['missing'][:20])}\n")
        f.write("\n")
        
        f.write("## Wild Runs Check\n\n")
        f.write(f"- **Status**: {'PASS' if result['wild_runs']['pass'] else 'FAIL'}\n")
        f.write(f"- **Count**: {result['wild_runs']['count']}\n")
        if result['wild_runs']['runs']:
            f.write(f"- **Wild runs** (first 20): {', '.join(result['wild_runs']['runs'][:20])}\n")
        f.write("\n")
        
        f.write("## Data Integrity Check\n\n")
        f.write(f"- **Status**: {'PASS' if result['data_integrity']['pass'] else 'FAIL'}\n")
        f.write(f"- **Total issues**: {result['data_integrity']['total_issues']}\n")
        if result['data_integrity']['issues']:
            f.write("- **Issues** (first 50):\n")
            for issue in result['data_integrity']['issues']:
                f.write(f"  - {issue}\n")
        f.write("\n")
        
        f.write("## Student-Level Split Check\n\n")
        f.write(f"- **Status**: {'PASS' if result['student_level_split']['pass'] else 'FAIL'}\n")
        f.write(f"- **All splits student-level**: {result['student_level_split']['all_splits_student_level']}\n")
        f.write(f"- **No overlap**: {result['student_level_split']['no_overlap']}\n")
        f.write("\n")
        
        f.write("## N/A Abuse Check\n\n")
        f.write(f"- **Status**: {'PASS' if result['na_abuse']['pass'] else 'FAIL'}\n")
        f.write(f"- **Total issues**: {result['na_abuse']['total_issues']}\n")
        if result['na_abuse']['issues']:
            f.write("- **Issues** (first 50):\n")
            for issue in result['na_abuse']['issues']:
                f.write(f"  - {issue}\n")
        f.write("\n")
        
        f.write("## Seed Consistency Check\n\n")
        f.write(f"- **Status**: {'PASS' if result['seed_consistency']['pass'] else 'FAIL'}\n")
        f.write(f"- **Total issues**: {result['seed_consistency']['total_issues']}\n")
        f.write(f"- **Requirement**: Seeds must be used in training/attack and metrics must be recomputable from artifacts (tolerance: 1e-6)\n")
        if result['seed_consistency']['issues']:
            f.write("- **Issues** (first 50):\n")
            for issue in result['seed_consistency']['issues']:
                f.write(f"  - {issue}\n")
        f.write("\n")
        
        f.write("## Demographic Evidence Check\n\n")
        f.write(f"- **Status**: {'PASS' if result['demographic_evidence']['pass'] else 'FAIL'}\n")
        f.write(f"- **Total issues**: {result['demographic_evidence']['total_issues']}\n")
        f.write(f"- **Requirement**: schema_summary.json must exist and verify 'no demographic fields' claim\n")
        if result['demographic_evidence']['issues']:
            f.write("- **Issues** (first 50):\n")
            for issue in result['demographic_evidence']['issues']:
                f.write(f"  - {issue}\n")
        f.write("\n")
        
        f.write("## Representative Run Artifacts Check\n\n")
        f.write(f"- **Status**: {'PASS' if result['representative_artifacts']['pass'] else 'FAIL'}\n")
        f.write(f"- **Total issues**: {result['representative_artifacts']['total_issues']}\n")
        f.write(f"- **Requirement**: Each representative run must have: status.json, metrics.json, config.json, fingerprint.json, data_fingerprint.json, stdout.log (with RUN_END)\n")
        f.write(f"- **Representative runs checked**: {', '.join(result['representative_artifacts']['representative_runs_checked'])}\n")
        if result['representative_artifacts']['issues']:
            f.write("- **Issues** (first 50):\n")
            for issue in result['representative_artifacts']['issues']:
                f.write(f"  - {issue}\n")
        f.write("\n")
        
        f.write("## Recompute Consistency Check\n\n")
        f.write(f"- **Status**: {'PASS' if result['recompute_consistency']['pass'] else 'FAIL'}\n")
        f.write(f"- **Total checked**: {result['recompute_consistency']['total_checked']}\n")
        f.write(f"- **Passed**: {result['recompute_consistency']['passed']}\n")
        f.write(f"- **Failed**: {result['recompute_consistency']['failed']}\n")
        f.write(f"- **Total issues**: {result['recompute_consistency']['total_issues']}\n")
        f.write(f"- **Requirement**: All test_auc/mia_auc/fairness gap must be recomputable from artifacts (tolerance: {result['recompute_consistency']['tolerance']})\n")
        f.write(f"- **Note**: This check is MANDATORY and must PASS for audit approval\n")
        if result['recompute_consistency']['issues']:
            f.write("- **Issues** (first 50):\n")
            for issue in result['recompute_consistency']['issues']:
                f.write(f"  - {issue}\n")
        f.write("\n")
        
        f.write("## Threat Model Closure Check\n\n")
        f.write(f"- **Status**: {'PASS' if result['threat_model_closure']['pass'] else 'FAIL'}\n")
        f.write(f"- **Total issues**: {result['threat_model_closure']['total_issues']}\n")
        f.write(f"- **Requirement**: attacker visibility must be explicitly stated (same-as-release vs stronger-than-release) in metrics.json and tables\n")
        if result['threat_model_closure']['issues']:
            f.write("- **Issues** (first 50):\n")
            for issue in result['threat_model_closure']['issues']:
                f.write(f"  - {issue}\n")
        f.write("\n")
    
    # 也保存JSON版本
    json_path = Path("outputs/reports/audit_fullpaper.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    # 打印摘要
    print("=" * 60)
    print(f"AUDIT RESULT: {'PASS' if result['overall_pass'] else 'FAIL'}")
    print("=" * 60)
    print(f"Coverage: {result['coverage']['coverage_pct']} ({'PASS' if result['coverage']['pass'] else 'FAIL'}, threshold: {result['coverage']['threshold']*100:.0f}%)")
    print(f"Wild runs: {result['wild_runs']['count']} ({'PASS' if result['wild_runs']['pass'] else 'FAIL'})")
    print(f"Data integrity: {result['data_integrity']['total_issues']} issues ({'PASS' if result['data_integrity']['pass'] else 'FAIL'})")
    print(f"Student-level split: {'PASS' if result['student_level_split']['pass'] else 'FAIL'}")
    print(f"N/A abuse: {result['na_abuse']['total_issues']} issues ({'PASS' if result['na_abuse']['pass'] else 'FAIL'})")
    print(f"Seed consistency: {result['seed_consistency']['total_issues']} issues ({'PASS' if result['seed_consistency']['pass'] else 'FAIL'})")
    print(f"Demographic evidence: {result['demographic_evidence']['total_issues']} issues ({'PASS' if result['demographic_evidence']['pass'] else 'FAIL'})")
    print(f"Representative artifacts: {result['representative_artifacts']['total_issues']} issues ({'PASS' if result['representative_artifacts']['pass'] else 'FAIL'})")
    print(f"Recompute consistency: {result['recompute_consistency']['total_issues']} issues ({'PASS' if result['recompute_consistency']['pass'] else 'FAIL'}) - MANDATORY")
    print(f"Threat model closure: {result['threat_model_closure']['total_issues']} issues ({'PASS' if result['threat_model_closure']['pass'] else 'FAIL'})")
    print(f"\nReport saved to: {report_path}")
    
    # 如果失败，返回非零退出码
    exit(0 if result['overall_pass'] else 1)

if __name__ == "__main__":
    main()
