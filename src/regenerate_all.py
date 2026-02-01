"""
重新生成所有输出：计划、实验、报告、审计、合理性检查
"""
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """运行命令并检查退出码"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, shell=True, capture_output=False)
    
    if result.returncode != 0:
        print(f"\n[FAILED] {description}")
        print(f"Exit code: {result.returncode}")
        return False
    else:
        print(f"\n[SUCCESS] {description}")
        return True

def main():
    """主函数：执行完整流程"""
    steps = [
        ("python src/generate_fast_plan.py", "Regenerate experiment plan (with negative controls)"),
        ("python src/run_all.py --only-plan outputs/reports/experiment_plan_fast.json --resume", "Re-run experiments (resume mode, only new runs)"),
        ("python src/reporting.py", "Regenerate all_tables.md"),
        ("python src/sanity_checks.py", "Run sanity checks"),
        ("python src/recompute_from_artifacts.py --plan outputs/reports/experiment_plan_fast.json", "Run recompute check"),
    ]
    
    print("="*60)
    print("REGENERATING ALL OUTPUTS")
    print("="*60)
    
    all_passed = True
    for cmd, desc in steps:
        if not run_command(cmd, desc):
            all_passed = False
            print(f"\n⚠️  Stopping due to failure in: {desc}")
            break
    
    if all_passed:
        print("\n" + "="*60)
        print("[SUCCESS] ALL STEPS COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nNext steps:")
        print("1. Review paper/sanity_report.md")
        print("2. Review outputs/reports/recompute_check.json")
        print("3. Generate paper/EDM_FULL_PAPER.md")
        print("4. Generate paper/CONTRIBUTIONS.md")
        print("5. Update paper/README.md")
    else:
        print("\n" + "="*60)
        print("[FAILED] PIPELINE FAILED")
        print("="*60)
        print("\nPlease review the errors above and fix them before continuing.")
        sys.exit(1)

if __name__ == "__main__":
    main()
