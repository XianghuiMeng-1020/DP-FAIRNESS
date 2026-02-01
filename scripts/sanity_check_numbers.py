#!/usr/bin/env python3
"""Number-level sanity checks for final submission hardening"""
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

base_dir = Path(__file__).parent.parent

def load_excluded_runs():
    """Load excluded runs"""
    excluded_path = base_dir / "paper" / "excluded_runs.json"
    if not excluded_path.exists():
        return set()
    try:
        with open(excluded_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return set(data.get("excluded_runs", []))
    except:
        return set()

def check_table12():
    """Check Table 12 negative controls"""
    all_tables_path = base_dir / "outputs" / "reports" / "all_tables.md"
    if not all_tables_path.exists():
        return False, "all_tables.md not found"
    
    content = all_tables_path.read_text(encoding='utf-8')
    
    # Find Table 12 Random Labels row
    # Pattern: | Random Labels | ... | Test AUC (mean ± CI) | ...
    pattern = r'\| Random Labels \|.*?\| ([0-9.]+) \[.*?\] \| ([0-9.]+) \[.*?\] \|'
    match = re.search(pattern, content)
    
    if not match:
        return False, "Table 12 Random Labels row not found"
    
    mia_auc_str = match.group(1)
    test_auc_str = match.group(2)
    
    try:
        mia_auc = float(mia_auc_str)
        test_auc = float(test_auc_str)
        
        # Check Test AUC ≈ 0.5
        tolerance = 0.1
        test_auc_ok = abs(test_auc - 0.5) <= tolerance
        
        # Check MIA AUC ≈ 0.5
        mia_auc_ok = abs(mia_auc - 0.5) <= tolerance
        
        if test_auc_ok and mia_auc_ok:
            return True, f"Random Labels: Test AUC = {test_auc:.5f} (≈0.5), MIA AUC = {mia_auc:.5f} (≈0.5)"
        else:
            return False, f"Random Labels: Test AUC = {test_auc:.5f} (diff={abs(test_auc-0.5):.5f}), MIA AUC = {mia_auc:.5f} (diff={abs(mia_auc-0.5):.5f})"
    except Exception as e:
        return False, f"Could not parse values: {e}"
    
    # Find Random Groups TPR Gap
    pattern_groups = r'\| Random Groups \|.*?\| ([0-9.]+) \[.*?\] \| ([0-9.]+) \[.*?\] \| ([0-9.]+) \[.*?\] \|'
    match_groups = re.search(pattern_groups, content)
    
    if match_groups:
        tpr_gap_str = match_groups.group(3)
        try:
            tpr_gap = float(tpr_gap_str)
            tolerance = 0.1
            if abs(tpr_gap) <= tolerance:
                return True, f"Random Groups: TPR Gap = {tpr_gap:.5f} (≈0)"
            else:
                return False, f"Random Groups: TPR Gap = {tpr_gap:.5f} (not ≈0)"
        except:
            pass
    
    return True, "Table 12 checks passed"

def check_coverage_counts():
    """Check coverage counts are consistent"""
    # Load plan
    plan_path = base_dir / "outputs" / "reports" / "experiment_plan_fast.json"
    if not plan_path.exists():
        return False, "experiment_plan_fast.json not found"
    
    with open(plan_path, 'r', encoding='utf-8') as f:
        plan = json.load(f)
    
    excluded_runs = load_excluded_runs()
    
    # Count expected runs (plan runs minus excluded)
    expected_runs = {entry["run_id"] for entry in plan}
    expected_count = len(expected_runs - excluded_runs)
    
    # Check audit report
    audit_path = base_dir / "outputs" / "reports" / "audit_fullpaper.md"
    if not audit_path.exists():
        return False, "audit_fullpaper.md not found"
    
    content = audit_path.read_text(encoding='utf-8')
    
    # Extract recompute counts
    recompute_match = re.search(r'Total checked.*?(\d+)', content)
    recompute_passed_match = re.search(r'Passed.*?(\d+)', content)
    
    if recompute_match and recompute_passed_match:
        recompute_total = int(recompute_match.group(1))
        recompute_passed = int(recompute_passed_match.group(1))
        
        # Extract coverage counts
        coverage_expected_match = re.search(r'Expected.*?(\d+)', content)
        coverage_ok_match = re.search(r'OK.*?(\d+)', content)
        
        if coverage_expected_match and coverage_ok_match:
            coverage_expected = int(coverage_expected_match.group(1))
            coverage_ok = int(coverage_ok_match.group(1))
            
            # Check consistency
            issues = []
            
            # Expected count should match plan minus excluded
            if coverage_expected != expected_count:
                issues.append(f"Coverage expected ({coverage_expected}) != plan minus excluded ({expected_count})")
            
            # Recompute total should match expected
            if recompute_total != expected_count:
                issues.append(f"Recompute total ({recompute_total}) != expected ({expected_count})")
            
            # Recompute should be 100% pass
            if recompute_passed != recompute_total:
                issues.append(f"Recompute passed ({recompute_passed}) != total ({recompute_total})")
            
            if issues:
                return False, "; ".join(issues)
            
            return True, f"Coverage: expected={coverage_expected}, ok={coverage_ok}, recompute={recompute_total}/{recompute_passed}"
    
    return False, "Could not extract counts from audit report"

def check_excluded_runs_not_in_tables():
    """Verify excluded runs are not referenced in tables"""
    excluded_runs = load_excluded_runs()
    
    all_tables_path = base_dir / "outputs" / "reports" / "all_tables.md"
    if not all_tables_path.exists():
        return False, "all_tables.md not found"
    
    content = all_tables_path.read_text(encoding='utf-8')
    
    found_excluded = []
    for run_id in excluded_runs:
        if run_id in content:
            found_excluded.append(run_id)
    
    if found_excluded:
        return False, f"Found excluded runs in tables: {found_excluded[:5]}"
    
    return True, f"All {len(excluded_runs)} excluded runs are not in tables"

def check_placeholder_count():
    """Check placeholder count is 0"""
    all_tables_path = base_dir / "outputs" / "reports" / "all_tables.md"
    if not all_tables_path.exists():
        return False, "all_tables.md not found"
    
    content = all_tables_path.read_text(encoding='utf-8')
    
    # Check placeholder count line
    placeholder_match = re.search(r'Placeholder count: (\d+)', content)
    if placeholder_match:
        count = int(placeholder_match.group(1))
        if count == 0:
            return True, "Placeholder count is 0"
        else:
            return False, f"Placeholder count is {count} (should be 0)"
    
    # Check for placeholder patterns
    placeholder_patterns = ["PENDING", "TODO", "TBD", "FILLME"]
    found = []
    for pattern in placeholder_patterns:
        if pattern in content:
            found.append(pattern)
    
    if found:
        return False, f"Found placeholder patterns: {found}"
    
    return True, "No placeholders found"

def main():
    print("=== Number-Level Sanity Checks ===\n")
    
    checks = [
        ("Table 12 Negative Controls", check_table12),
        ("Coverage Counts Consistency", check_coverage_counts),
        ("Excluded Runs Not in Tables", check_excluded_runs_not_in_tables),
        ("Placeholder Count", check_placeholder_count),
    ]
    
    results = []
    all_pass = True
    
    for name, check_func in checks:
        passed, message = check_func()
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
        print(f"  {message}\n")
        results.append({"name": name, "pass": passed, "message": message})
        if not passed:
            all_pass = False
    
    print(f"\n=== Overall Result: {'✅ PASS' if all_pass else '❌ FAIL'} ===")
    
    return all_pass

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
