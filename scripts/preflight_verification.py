"""
STEP 4: Preflight verification BEFORE rerun
Verify we are truly using real data by checking file existence, successful parsing, and plausible sample sizes
"""
import sys
import io
from pathlib import Path
import pandas as pd
import numpy as np

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_dataset

def verify_dataset(dataset_name: str, seed: int = 42):
    """Verify a dataset uses real data"""
    print(f"\n{'='*80}")
    print(f"Verifying {dataset_name}")
    print('='*80)
    
    # Check file existence
    data_dir = Path("data")
    file_paths = []
    
    if dataset_name == "OULAD":
        possible_paths = [
            data_dir / "raw" / "oulad" / "studentInfo.csv",
            data_dir / "OULAD" / "studentInfo.csv",
        ]
        for base in [data_dir / "raw" / "oulad", data_dir / "OULAD"]:
            if base.exists():
                for csv_file in base.rglob("studentInfo.csv"):
                    possible_paths.append(csv_file)
                    break
    elif dataset_name == "UCI697":
        possible_paths = [
            data_dir / "raw" / "uci697" / "data.csv",
            data_dir / "raw" / "uci697" / "student-mat.csv",
            data_dir / "UCI697" / "student-mat.csv",
        ]
    elif dataset_name == "HarvardX_PersonCourse":
        possible_paths = [
            data_dir / "raw" / "harvardx" / "HXPC13_DI_v3_11-13-2019.tab",
            data_dir / "raw" / "harvardx" / "HMXPC13_DI_v2_5-14-14.csv",
            data_dir / "HarvardX_PersonCourse" / "HMXPC13_DI_v2_5-14-14.csv",
        ]
    
    found_file = None
    for path in possible_paths:
        if path.exists():
            found_file = path
            break
    
    if not found_file:
        print(f"ERROR: No data file found for {dataset_name}")
        print(f"Searched paths: {[str(p) for p in possible_paths]}")
        return False
    
    print(f"Found data file: {found_file}")
    file_size = found_file.stat().st_size
    print(f"File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
    
    # Load dataset
    try:
        X_train, X_test, y_train, y_test, groups_test = load_dataset(
            dataset_name, seed=seed, data_dir=str(data_dir)
        )
        
        n_total = len(X_train) + len(X_test)
        n_train = len(X_train)
        n_test = len(X_test)
        
        print(f"\nDataset statistics:")
        print(f"  n_total: {n_total:,}")
        print(f"  n_train: {n_train:,}")
        print(f"  n_test: {n_test:,}")
        print(f"  n_features: {X_train.shape[1]}")
        
        # Sanity checks
        print(f"\nSanity checks:")
        
        # Check if sizes are plausible (not synthetic-like fixed sizes)
        if dataset_name == "OULAD":
            if n_total == 5000:
                print(f"  ERROR: n_total=5000 suggests synthetic data (real OULAD should be >>5000)")
                return False
            elif n_total < 10000:
                print(f"  WARNING: n_total={n_total} seems small for OULAD")
            else:
                print(f"  PASS: n_total={n_total} is plausible for OULAD")
        
        elif dataset_name == "UCI697":
            if n_total == 400:
                print(f"  ERROR: n_total=400 suggests synthetic data (real UCI697 should be ~697)")
                return False
            elif n_total < 300 or n_total > 1000:
                print(f"  WARNING: n_total={n_total} seems unusual for UCI697 (expected ~697)")
            else:
                print(f"  PASS: n_total={n_total} is plausible for UCI697")
        
        elif dataset_name == "HarvardX_PersonCourse":
            if n_total == 3000:
                print(f"  ERROR: n_total=3000 suggests synthetic data (real HarvardX should be >>3000)")
                return False
            elif n_total < 10000:
                print(f"  WARNING: n_total={n_total} seems small for HarvardX")
            else:
                print(f"  PASS: n_total={n_total} is plausible for HarvardX")
        
        # Check that n_train is not fixed at 4000
        if n_train == 4000:
            print(f"  ERROR: n_train=4000 is suspiciously fixed (suggests synthetic)")
            return False
        else:
            print(f"  PASS: n_train={n_train} varies with dataset scale")
        
        # Check label distribution
        unique_labels_train = np.unique(y_train)
        unique_labels_test = np.unique(y_test)
        print(f"\nLabel distribution:")
        print(f"  Train labels: {unique_labels_train}")
        print(f"  Test labels: {unique_labels_test}")
        print(f"  Train label counts: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        print(f"  Test label counts: {dict(zip(*np.unique(y_test, return_counts=True)))}")
        
        if len(unique_labels_train) < 2 or len(unique_labels_test) < 2:
            print(f"  WARNING: Binary classification requires both classes in train and test")
        
        print(f"\nSUCCESS: {dataset_name} verification passed")
        return True
        
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main verification function"""
    print("="*80)
    print("STEP 4: Preflight Verification BEFORE Rerun")
    print("="*80)
    print("\nVerifying that we are truly using real data...")
    
    datasets = ["OULAD", "UCI697", "HarvardX_PersonCourse"]
    results = {}
    
    for dataset in datasets:
        results[dataset] = verify_dataset(dataset)
    
    print("\n" + "="*80)
    print("Verification Summary:")
    print("="*80)
    for dataset, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"{dataset:<30} {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n" + "="*80)
        print("SUCCESS: All datasets verified. Ready for rerun!")
        print("="*80)
        return True
    else:
        print("\n" + "="*80)
        print("ERROR: Some datasets failed verification. DO NOT proceed with rerun.")
        print("="*80)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
