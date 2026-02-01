"""
生成计划统计报告
"""
import json
from collections import Counter
from pathlib import Path

def main():
    plan_path = Path("outputs/reports/experiment_plan_fast.json")
    plan = json.load(open(plan_path, "r", encoding="utf-8"))
    
    print("=" * 60)
    print("PLAN STATISTICS (Reviewer-Proof Fast)")
    print("=" * 60)
    print(f"\nTotal runs: {len(plan)} (target: 180-260)")
    print(f"Core runs (seeds=5, main tables): {sum(1 for e in plan if e['is_core'])}")
    print(f"Diagnostic runs (seeds=2, appendix only): {sum(1 for e in plan if e['is_diagnostic'])}")
    
    print("\n" + "=" * 60)
    print("DATASET COVERAGE")
    print("=" * 60)
    ds = Counter(e['dataset'] for e in plan)
    for d, c in sorted(ds.items()):
        print(f"  {d}: {c} runs")
    
    print("\n" + "=" * 60)
    print("MODEL COVERAGE (per dataset)")
    print("=" * 60)
    mod = Counter((e['dataset'], e['model'], e.get('model_variant')) for e in plan)
    for (d, m, v), c in sorted(mod.items()):
        variant_str = v if v else "N/A"
        print(f"  {d} x {m} ({variant_str}): {c} runs")
    
    print("\n" + "=" * 60)
    print("DEFENSE COVERAGE (sample)")
    print("=" * 60)
    defs = Counter((e['dataset'], e['train_defense'], e.get('publish_defense') or 'none', e.get('eps')) for e in plan)
    for (d, td, pd, eps), c in sorted(defs.items())[:30]:
        eps_str = str(eps) if eps else "N/A"
        print(f"  {d} x {td} x {pd} x eps={eps_str}: {c} runs")
    
    print("\n" + "=" * 60)
    print("FAIRNESS COVERAGE (OULAD)")
    print("=" * 60)
    fairness = Counter((e['dataset'], e['fairness_attribute']) for e in plan if e['dataset'] == 'OULAD')
    for (d, f), c in sorted(fairness.items()):
        print(f"  {d} x {f}: {c} runs")
    
    print("\n" + "=" * 60)
    print("SEEDS CONSISTENCY CHECK")
    print("=" * 60)
    # 检查每个配置的seeds数量
    config_seeds = {}
    for e in plan:
        key = (e['dataset'], e['model'], e.get('model_variant'), e['train_defense'], 
               e.get('publish_defense') or 'none', e.get('eps'), e['visibility'], e['fairness_attribute'])
        if key not in config_seeds:
            config_seeds[key] = []
        config_seeds[key].append(e['seed'])
    
    inconsistent = []
    for key, seeds_list in config_seeds.items():
        # 找到对应的entry来检查is_core/is_diagnostic
        sample_entry = next((e for e in plan if (e['dataset'], e['model'], e.get('model_variant'), e['train_defense'], 
                                                  e.get('publish_defense') or 'none', e.get('eps'), e['visibility'], e['fairness_attribute']) == key), None)
        if sample_entry:
            if sample_entry['is_core']:
                if len(set(seeds_list)) != 5 or set(seeds_list) != {1, 2, 3, 4, 5}:
                    inconsistent.append((key, seeds_list))
            elif sample_entry['is_diagnostic']:
                if len(set(seeds_list)) != 2 or set(seeds_list) != {1, 2}:
                    inconsistent.append((key, seeds_list))
    
    if inconsistent:
        print(f"  WARNING: Found {len(inconsistent)} inconsistent seed configurations")
        for key, seeds in inconsistent[:5]:
            print(f"    {key}: seeds={seeds}")
    else:
        print("  ✓ All configurations have consistent seeds")
    
    print("\n" + "=" * 60)
    print("COVERAGE SUFFICIENCY ASSESSMENT")
    print("=" * 60)
    print(f"  Datasets: {len(ds)} (required: 3) {'✓' if len(ds) >= 3 else '✗'}")
    print(f"  Models per dataset: min={min(Counter(e['model'] for e in plan if e['dataset'] == d).values() for d in ds)} (required: ≥4) {'✓' if min(Counter(e['model'] for e in plan if e['dataset'] == d).values() for d in ds) >= 4 else '✗'}")
    print(f"  OULAD fairness attrs: {len([f for f in set(e['fairness_attribute'] for e in plan if e['dataset'] == 'OULAD') if f != 'NA'])} (required: ≥4) {'✓' if len([f for f in set(e['fairness_attribute'] for e in plan if e['dataset'] == 'OULAD') if f != 'NA']) >= 4 else '✗'}")
    print(f"  DP-SGD ε values: {len(set(e.get('eps') for e in plan if e['train_defense'] == 'DP-SGD' and e.get('eps')))}) (required: 3) {'✓' if len(set(e.get('eps') for e in plan if e['train_defense'] == 'DP-SGD' and e.get('eps'))) >= 3 else '✗'}")
    print(f"  Release-time defenses: {len(set(e.get('publish_defense') for e in plan if e.get('publish_defense')))}) (required: ≥2) {'✓' if len(set(e.get('publish_defense') for e in plan if e.get('publish_defense'))) >= 2 else '✗'}")

if __name__ == "__main__":
    main()
