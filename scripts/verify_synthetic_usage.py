"""
STEP 0: Verify whether current runs used synthetic fallback
Check actual usage for each dataset
"""
import sys
import io
from pathlib import Path
import json
import pandas as pd
import numpy as np

# Fix Windows encoding issues
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_dataset

def check_dataset_status(dataset_name: str, seed: int = 42):
    """Check if dataset uses synthetic data"""
    # 检查数据文件是否存在
    data_dir = None
    possible_dirs = [Path("data"), Path("datasets"), Path("outputs/data")]
    for d in possible_dirs:
        if d.exists():
            data_dir = str(d)
            break
    
    # 根据数据集检查文件路径
    if dataset_name == "OULAD":
        expected_path = Path(data_dir) / "OULAD" / "studentInfo.csv" if data_dir else None
        file_exists = expected_path.exists() if expected_path else False
    elif dataset_name == "UCI697":
        expected_path = Path(data_dir) / "UCI697" / "student-mat.csv" if data_dir else None
        file_exists = expected_path.exists() if expected_path else False
    elif dataset_name == "HarvardX_PersonCourse":
        expected_path = Path(data_dir) / "HarvardX_PersonCourse" / "HMXPC13_DI_v2_5-14-14.csv" if data_dir else None
        file_exists = expected_path.exists() if expected_path else False
    else:
        return None
    
    # 尝试加载数据集（这会触发合成数据生成如果文件不存在）
    try:
        X_train, X_test, y_train, y_test, groups_test = load_dataset(dataset_name, seed=seed, data_dir=data_dir)
        n_total = len(X_train) + len(X_test)
        n_train = len(X_train)
        n_test = len(X_test)
        
        # 检查是否是合成数据（通过样本数量判断）
        is_synthetic = False
        evidence = ""
        
        if dataset_name == "OULAD":
            # 合成数据固定为5000样本
            if n_total == 5000:
                is_synthetic = True
                evidence = f"Synthetic: fixed n_total=5000 (real OULAD should be >>5000)"
            elif not file_exists:
                is_synthetic = True
                evidence = f"Synthetic fallback: file not found at {expected_path}"
            else:
                evidence = f"Real data: file={expected_path}, n_total={n_total}"
        elif dataset_name == "UCI697":
            # 合成数据固定为400样本
            if n_total == 400:
                is_synthetic = True
                evidence = f"Synthetic: fixed n_total=400 (real UCI697 should be ~697)"
            elif not file_exists:
                is_synthetic = True
                evidence = f"Synthetic fallback: file not found at {expected_path}"
            else:
                evidence = f"Real data: file={expected_path}, n_total={n_total}"
        elif dataset_name == "HarvardX_PersonCourse":
            # 合成数据固定为3000样本
            if n_total == 3000:
                is_synthetic = True
                evidence = f"Synthetic: fixed n_total=3000 (real HarvardX should be >>3000)"
            elif not file_exists:
                is_synthetic = True
                evidence = f"Synthetic fallback: file not found at {expected_path}"
            else:
                evidence = f"Real data: file={expected_path}, n_total={n_total}"
        
        return {
            "dataset_name": dataset_name,
            "is_synthetic": is_synthetic,
            "n_total_used": n_total,
            "n_train_used": n_train,
            "n_test_used": n_test,
            "evidence": evidence,
            "file_exists": file_exists,
            "expected_path": str(expected_path) if expected_path else "N/A"
        }
    except Exception as e:
        return {
            "dataset_name": dataset_name,
            "is_synthetic": None,
            "n_total_used": None,
            "n_train_used": None,
            "n_test_used": None,
            "evidence": f"Error loading: {str(e)}",
            "file_exists": file_exists,
            "expected_path": str(expected_path) if expected_path else "N/A"
        }

def main():
    """Main function: check all datasets"""
    datasets = ["OULAD", "UCI697", "HarvardX_PersonCourse"]
    
    print("=" * 80)
    print("STEP 0: Verify whether current runs used synthetic fallback")
    print("=" * 80)
    print()
    
    results = []
    for dataset in datasets:
        print(f"Checking {dataset}...")
        result = check_dataset_status(dataset)
        results.append(result)
    
    # Print table
    print()
    print("=" * 80)
    print("Verification Results Table:")
    print("=" * 80)
    print(f"{'Dataset':<25} {'Is Synthetic':<15} {'n_total':<12} {'n_train':<12} {'n_test':<12} {'Evidence'}")
    print("-" * 80)
    
    any_synthetic = False
    for r in results:
        is_syn = "YES" if r["is_synthetic"] else ("NO" if r["is_synthetic"] is False else "UNKNOWN")
        n_total = str(r["n_total_used"]) if r["n_total_used"] is not None else "N/A"
        n_train = str(r["n_train_used"]) if r["n_train_used"] is not None else "N/A"
        n_test = str(r["n_test_used"]) if r["n_test_used"] is not None else "N/A"
        evidence = r["evidence"][:50] + "..." if len(r["evidence"]) > 50 else r["evidence"]
        
        print(f"{r['dataset_name']:<25} {is_syn:<15} {n_total:<12} {n_train:<12} {n_test:<12} {evidence}")
        
        if r["is_synthetic"]:
            any_synthetic = True
    
    print()
    print("=" * 80)
    if any_synthetic:
        print("WARNING: Synthetic fallback detected!")
        print()
        print("Details:")
        for r in results:
            if r["is_synthetic"]:
                print(f"\nDataset: {r['dataset_name']}")
                print(f"  Reason: {r['evidence']}")
                print(f"  Expected path: {r['expected_path']}")
                print(f"  File exists: {r['file_exists']}")
        print()
        print("Please download real datasets first, then re-run this script to verify.")
        return False
    else:
        print("SUCCESS: All datasets use real data")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
