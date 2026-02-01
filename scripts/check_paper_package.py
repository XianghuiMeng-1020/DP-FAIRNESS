#!/usr/bin/env python3
"""Final paper package verification script"""
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

base_dir = Path(__file__).parent.parent
paper_dir = base_dir / "paper"

def check_required_files():
    """Check all required files exist in paper/"""
    required_files = [
        "all_tables.md",
        "audit_fullpaper.md",
        "sanity_report.md",
        "contamination_report.md",
        "README.md",
        "DELIVERY_SUMMARY.md",
        "CONTRIBUTIONS.md",
        "excluded_runs.json",
        "representative_runs_hashes.json",
    ]
    
    missing = []
    for filename in required_files:
        if not (paper_dir / filename).exists():
            missing.append(filename)
    
    if missing:
        return False, f"Missing files: {missing}"
    return True, f"All {len(required_files)} required files exist"

def check_placeholders():
    """Check placeholder count is 0"""
    all_tables_path = paper_dir / "all_tables.md"
    if not all_tables_path.exists():
        return False, "all_tables.md not found"
    
    content = all_tables_path.read_text(encoding='utf-8')
    
    # Check placeholder count
    placeholder_match = re.search(r'Placeholder count: (\d+)', content)
    if placeholder_match:
        count = int(placeholder_match.group(1))
        if count != 0:
            return False, f"Placeholder count is {count} (should be 0)"
    
    # Check for placeholder patterns
    placeholder_patterns = ["PENDING", "TODO", "TBD", "FILLME", "PLACEHOLDER"]
    found = []
    for pattern in placeholder_patterns:
        if pattern in content and "Placeholder count: 0" not in content:
            # Check if it's in a data cell
            lines = content.split('\n')
            for line in lines:
                if '|' in line and pattern in line and 'count' not in line.lower():
                    found.append(f"{pattern} in line: {line[:80]}")
                    break
    
    if found:
        return False, f"Found placeholder patterns: {found[:3]}"
    
    return True, "Placeholder count is 0, no placeholder patterns found"

def check_table12_auc():
    """Check Table 12 contains corrected Test AUC ~0.5"""
    all_tables_path = paper_dir / "all_tables.md"
    if not all_tables_path.exists():
        return False, "all_tables.md not found"
    
    content = all_tables_path.read_text(encoding='utf-8')
    
    # Find Table 12 Random Labels row
    pattern = r'\| Random Labels \|.*?\| ([0-9.]+) \[.*?\] \| ([0-9.]+) \[.*?\] \|'
    match = re.search(pattern, content)
    
    if not match:
        return False, "Table 12 Random Labels row not found"
    
    mia_auc_str = match.group(1)
    test_auc_str = match.group(2)
    
    try:
        test_auc = float(test_auc_str)
        tolerance = 0.1
        if abs(test_auc - 0.5) <= tolerance:
            return True, f"Table 12 Random Labels Test AUC = {test_auc:.5f} (≈0.5)"
        else:
            return False, f"Table 12 Random Labels Test AUC = {test_auc:.5f} (not ≈0.5, diff={abs(test_auc-0.5):.5f})"
    except Exception as e:
        return False, f"Could not parse Test AUC: {e}"

def check_audit_recompute():
    """Check audit recompute PASS count matches acceptance summary"""
    audit_path = paper_dir / "audit_fullpaper.md"
    if not audit_path.exists():
        return False, "audit_fullpaper.md not found"
    
    content = audit_path.read_text(encoding='utf-8')
    
    # Extract recompute counts
    recompute_match = re.search(r'Total checked.*?(\d+)', content)
    recompute_passed_match = re.search(r'Passed.*?(\d+)', content)
    recompute_failed_match = re.search(r'Failed.*?(\d+)', content)
    
    if not recompute_match or not recompute_passed_match:
        return False, "Could not extract recompute counts from audit"
    
    total = int(recompute_match.group(1))
    passed = int(recompute_passed_match.group(1))
    failed = int(recompute_failed_match.group(1)) if recompute_failed_match else 0
    
    # Check status
    if "Recompute Consistency Check" in content:
        recompute_section = content.split("Recompute Consistency Check")[1].split("##")[0]
        if "**Status**: PASS" not in recompute_section and "**Status**: ✅ PASS" not in recompute_section:
            return False, f"Recompute consistency status is not PASS"
    
    if failed != 0:
        return False, f"Recompute has {failed} failures (should be 0)"
    
    if passed != total:
        return False, f"Recompute passed {passed}/{total} (should be {total}/{total})"
    
    return True, f"Recompute consistency: {passed}/{total} PASS (100% coverage)"

def check_excluded_runs():
    """Check excluded_runs.json exists and is referenced"""
    excluded_path = paper_dir / "excluded_runs.json"
    if not excluded_path.exists():
        return False, "excluded_runs.json not found"
    
    try:
        with open(excluded_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            excluded_count = len(data.get("excluded_runs", []))
    except Exception as e:
        return False, f"Could not read excluded_runs.json: {e}"
    
    # Check README references it
    readme_path = paper_dir / "README.md"
    if readme_path.exists():
        readme_content = readme_path.read_text(encoding='utf-8')
        if "excluded_runs.json" not in readme_content:
            return False, "README.md does not reference excluded_runs.json"
    
    # Check DELIVERY_SUMMARY references it
    summary_path = paper_dir / "DELIVERY_SUMMARY.md"
    if summary_path.exists():
        summary_content = summary_path.read_text(encoding='utf-8')
        if "excluded_runs.json" not in summary_content:
            return False, "DELIVERY_SUMMARY.md does not reference excluded_runs.json"
    
    return True, f"excluded_runs.json exists with {excluded_count} excluded runs, referenced in README and DELIVERY_SUMMARY"

def main():
    print("=== Paper Package Verification ===\n")
    
    checks = [
        ("Required files exist", check_required_files),
        ("Placeholder count is 0", check_placeholders),
        ("Table 12 contains corrected Test AUC ~0.5", check_table12_auc),
        ("Audit recompute PASS count matches", check_audit_recompute),
        ("excluded_runs.json exists and referenced", check_excluded_runs),
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
