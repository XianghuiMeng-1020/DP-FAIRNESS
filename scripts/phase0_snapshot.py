#!/usr/bin/env python3
"""PHASE 0: Snapshot current state"""
import json
import os
import sys
from pathlib import Path

# Set UTF-8 encoding for output
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

base_dir = Path(__file__).parent.parent

# 1. Check key files
print("=== PHASE 0: Snapshot Current State ===\n")
print("1. Key file paths:")
files_to_check = [
    "paper/all_tables.md",
    "paper/audit_fullpaper.md", 
    "paper/sanity_report.md",
    "paper/EXECUTION_COMPLETE.md"
]

for f in files_to_check:
    path = base_dir / f
    status = "EXISTS" if path.exists() else "MISSING"
    print(f"  - {f}: {status}")

# 2. Extract Table 12 current values
print("\n2. Table 12 current state:")
all_tables_path = base_dir / "paper/all_tables.md"
if all_tables_path.exists():
    content = all_tables_path.read_text(encoding='utf-8')
    # 查找Table 12
    if "Table 12:" in content:
        lines = content.split('\n')
        in_table12 = False
        for i, line in enumerate(lines):
            if "Table 12:" in line:
                in_table12 = True
            if in_table12 and "| Random Labels |" in line:
                print(f"  Random Labels行: {line.strip()}")
                # 提取Test AUC值
                parts = [p.strip() for p in line.split('|')]
                if len(parts) > 5:
                    test_auc_str = parts[5]  # Test AUC列
                    print(f"    Current Test AUC: {test_auc_str}")
                    if "1.00000" in test_auc_str:
                        print("    WARNING: Test AUC = 1.0 (should be ~0.5)")
            if in_table12 and "| Random Groups |" in line:
                print(f"  Random Groups行: {line.strip()}")

# 3. Identify run_ids used by Table 12
print("\n3. Run IDs used by Table 12:")
plan_path = base_dir / "outputs/reports/experiment_plan_fast.json"
if plan_path.exists():
    with open(plan_path, 'r', encoding='utf-8') as f:
        plan = json.load(f)
    
    random_labels_runs = []
    random_groups_runs = []
    
    for entry in plan:
        run_id = entry.get("run_id", "")
        if "random_labels" in run_id.lower():
            random_labels_runs.append(run_id)
        elif "random_groups" in run_id.lower():
            random_groups_runs.append(run_id)
    
    print(f"  Random Labels runs: {len(random_labels_runs)} total")
    print(f"    示例: {random_labels_runs[:3]}")
    print(f"  Random Groups runs: {len(random_groups_runs)} total")
    print(f"    示例: {random_groups_runs[:3]}")

print("\n=== PHASE 0 Complete ===")
