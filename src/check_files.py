"""检查所有必需文件"""
import os
from pathlib import Path

files = [
    'audit_fullpaper.md',
    'all_tables.md',
    'contamination_report.md',
    'plan_execution_summary.json',
    'experiment_plan_fast.json',
    'EVIDENCE_PACKAGE.md'
]

base = Path('outputs/reports')
print('Files check:')
for f in files:
    path = base / f
    status = "EXISTS" if path.exists() else "MISSING"
    print(f'  {f}: {status}')
