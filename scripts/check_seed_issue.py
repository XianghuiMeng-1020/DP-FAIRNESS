"""
Check seed consistency issue for OULAD|MLP|DP-SGD|output_coarsening|eps=5
"""
import json
from pathlib import Path

# Load core_seed_metrics
with open("outputs/reports/core_seed_metrics_long.json", encoding='utf-8') as f:
    data = json.load(f)

# Find the problematic setting
for setting_data in data.get("settings", []):
    setting = setting_data["setting"]
    if (setting.get('dataset') == 'OULAD' and 
        setting.get('model') == 'MLP' and 
        setting.get('train_defense') == 'DP-SGD' and
        setting.get('publish_defense') == 'output_coarsening' and
        setting.get('eps') == 5):
        
        print(f"Found problematic setting:")
        print(f"  Dataset: {setting.get('dataset')}")
        print(f"  Model: {setting.get('model')}")
        print(f"  Train Defense: {setting.get('train_defense')}")
        print(f"  Publish Defense: {setting.get('publish_defense')}")
        print(f"  Eps: {setting.get('eps')}")
        print(f"\nSeed metrics:")
        
        seeds = setting_data.get("seeds", [])
        for seed_row in seeds:
            run_id = seed_row.get("run_id")
            seed = seed_row.get("seed")
            test_auc = seed_row.get("test_auc")
            test_f1 = seed_row.get("test_f1")
            ece = seed_row.get("ece")
            worst_group_tpr_gap = seed_row.get("worst_group_tpr_gap")
            
            print(f"\n  Seed {seed} (run_id: {run_id}):")
            print(f"    test_auc: {test_auc}")
            print(f"    test_f1: {test_f1}")
            print(f"    ece: {ece}")
            print(f"    worst_group_tpr_gap: {worst_group_tpr_gap}")
        
        # Check if values are identical
        test_aucs = [s.get("test_auc") for s in seeds if s.get("test_auc") is not None]
        if len(set(test_aucs)) == 1:
            print(f"\n  WARNING: All test_auc values are identical: {test_aucs[0]}")
        
        break
