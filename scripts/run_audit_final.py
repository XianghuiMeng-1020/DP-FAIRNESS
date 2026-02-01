"""运行最终audit并生成报告"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.audit_fullpaper import audit_strict
import json

result = audit_strict('outputs/reports/experiment_plan_fast.json')

print("=== AUDIT RESULTS ===")
print(f"Coverage: {result['coverage']['coverage_pct']} ({result['coverage']['ok']}/{result['coverage']['expected']})")
wild_runs_list = list(result['wild_runs']) if isinstance(result['wild_runs'], (set, dict)) else result['wild_runs']
print(f"Wild runs: {len(wild_runs_list)}")
if len(wild_runs_list) > 0:
    print(f"  First 5: {wild_runs_list[:5]}")
print(f"Missing runs: {len(result['coverage']['missing'])}")
if len(result['coverage']['missing']) > 0:
    print(f"  First 5: {result['coverage']['missing'][:5]}")

# 保存到paper/audit_fullpaper.md
output_path = Path("paper/audit_fullpaper.md")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    f.write("# Full Paper Audit Report\n\n")
    
    # Handle wild_runs structure - it's a dict with 'pass', 'count', 'runs' keys
    wild_runs_dict = result['wild_runs']
    wild_runs_list = wild_runs_dict.get('runs', [])
    wild_runs_count = wild_runs_dict.get('count', len(wild_runs_list))
    
    # 判断总体状态
    coverage_ok = result['coverage']['coverage'] >= 0.95
    wild_runs_ok = wild_runs_count == 0
    overall_status = "PASS" if (coverage_ok and wild_runs_ok) else "FAIL"
    
    f.write(f"**OVERALL STATUS: {overall_status}**\n\n")
    
    f.write("## Coverage Check\n\n")
    f.write(f"- **Status**: {'PASS' if coverage_ok else 'FAIL'}\n")
    f.write(f"- **Expected**: {result['coverage']['expected']}\n")
    f.write(f"- **OK**: {result['coverage']['ok']}\n")
    f.write(f"- **Coverage**: {result['coverage']['coverage_pct']} (threshold: 95%)\n")
    missing_list = list(result['coverage']['missing']) if isinstance(result['coverage']['missing'], (set, dict)) else result['coverage']['missing']
    if len(missing_list) > 0:
        f.write(f"- **Missing runs** (first 20): {', '.join(missing_list[:20])}\n")
    f.write("\n")
    
    f.write("## Wild Runs Check\n\n")
    f.write(f"- **Status**: {'PASS' if wild_runs_ok else 'FAIL'}\n")
    f.write(f"- **Count**: {wild_runs_count}\n")
    if len(wild_runs_list) > 0:
        f.write(f"- **Wild runs** (first 20): {', '.join(str(r) for r in wild_runs_list[:20])}\n")
    f.write("\n")

print(f"\nSaved audit report to {output_path}")
