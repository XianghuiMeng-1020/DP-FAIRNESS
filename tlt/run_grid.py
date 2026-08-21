"""Build the experiment manifest and execute runs with resume."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from tlt.protocol import load_protocol, train_seed
from tlt.run_one import execute_run, is_complete, run_dir
from tlt.seeds import write_seed_plan

ROOT = Path(__file__).resolve().parents[1]


def secondary_only_cells():
    return secondary_cells()


def primary_cells():
    return [
        ("OULAD", "day28"),
        ("UCI697", "semester1"),
    ]


def secondary_cells():
    return [
        ("OULAD", "day14"),
        ("OULAD", "day56"),
        ("UCI697", "enrollment"),
    ]


def build_manifest(include_secondary: bool = False, smoke: bool = False) -> List[Dict]:
    proto = load_protocol()
    reps = [0] if smoke else list(proto["seeds"]["replicate_ids"])
    cells = secondary_only_cells() if include_secondary else primary_cells()
    archs = ["logistic", "mlp_small"]
    rows: List[Dict] = []

    def add(**kwargs):
        r = dict(kwargs)
        r["split_seed"] = proto["split_seed"]
        r["run_id"] = (
            f"{r['dataset']}_{r['checkpoint']}_{r['architecture']}_{r['condition']}_r{r['replicate']:02d}"
        )
        rows.append(r)

    for dataset, checkpoint in cells:
        for arch in archs:
            for r in reps:
                add(
                    dataset=dataset,
                    checkpoint=checkpoint,
                    architecture=arch,
                    condition="NP_A",
                    mechanism="none",
                    epsilon=None,
                    weight_decay=proto["optimization"]["weight_decay_default"],
                    replicate=r,
                    train_seed=train_seed("NP_A", r),
                    family="primary",
                )
                add(
                    dataset=dataset,
                    checkpoint=checkpoint,
                    architecture=arch,
                    condition="NP_B",
                    mechanism="none",
                    epsilon=None,
                    weight_decay=proto["optimization"]["weight_decay_default"],
                    replicate=r,
                    train_seed=train_seed("NP_B", r),
                    family="primary",
                )
                for eps in proto["privacy"]["target_epsilons"]:
                    add(
                        dataset=dataset,
                        checkpoint=checkpoint,
                        architecture=arch,
                        condition=f"DP_eps{int(eps)}",
                        mechanism="DP-SGD",
                        epsilon=float(eps),
                        weight_decay=proto["optimization"]["weight_decay_default"],
                        replicate=r,
                        train_seed=train_seed("DP", r),
                        family="primary",
                    )
                for wd in proto["optimization"]["specification_weight_decay_grid"]:
                    if float(wd) == float(proto["optimization"]["weight_decay_default"]):
                        continue
                    add(
                        dataset=dataset,
                        checkpoint=checkpoint,
                        architecture=arch,
                        condition=f"NP_WD_{wd}",
                        mechanism="none",
                        epsilon=None,
                        weight_decay=float(wd),
                        replicate=r,
                        train_seed=train_seed("NP_GRID", r),
                        family="specification_grid",
                    )
    return rows


def write_manifest(rows: List[Dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "run_id",
        "dataset",
        "checkpoint",
        "architecture",
        "condition",
        "mechanism",
        "epsilon",
        "weight_decay",
        "replicate",
        "train_seed",
        "split_seed",
        "family",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})
    return path


def append_exclusion(run_id: str, reason: str, attempt: int) -> None:
    path = ROOT / "artifacts" / "manifests" / "run_exclusions.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["run_id", "reason", "attempt", "action"])
        if not exists:
            w.writeheader()
        w.writerow({"run_id": run_id, "reason": reason, "attempt": attempt, "action": "retry_then_log"})


def execute_manifest(rows: List[Dict], max_retries: int = 1) -> None:
    excl = ROOT / "artifacts" / "manifests" / "run_exclusions.csv"
    if not excl.exists():
        with open(excl, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=["run_id", "reason", "attempt", "action"]).writeheader()
    n = len(rows)
    for i, entry in enumerate(rows, 1):
        rid = entry["run_id"]
        print(f"[{i}/{n}] {rid}", flush=True)
        info = execute_run(entry)
        if info.get("status") == "failed":
            append_exclusion(rid, info.get("reason", "runtime_failure"), 1)
            print(f"  FAIL {info.get('reason')}; retrying identical config", flush=True)
            info = execute_run(entry, overwrite=True)
            if info.get("status") == "failed":
                append_exclusion(rid, info.get("reason", "runtime_failure"), 2)
                print(f"  FAIL again; logged exclusion", flush=True)
            else:
                print(f"  retry ok", flush=True)
        else:
            print(f"  {info.get('status', 'ok')} val_auc={info.get('val_auc')}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--secondary", action="store_true")
    parser.add_argument("--validate-seeds", type=int, default=0, help="first N seeds only")
    args = parser.parse_args()
    write_seed_plan()
    rows = build_manifest(include_secondary=args.secondary, smoke=args.smoke)
    if args.validate_seeds > 0:
        rows = [r for r in rows if r["replicate"] < args.validate_seeds]
    if args.smoke:
        man_name = "experiment_manifest_smoke.csv"
    elif args.secondary:
        man_name = "experiment_manifest_secondary.csv"
    else:
        man_name = "experiment_manifest.csv"
    man = ROOT / "artifacts" / "manifests" / man_name
    write_manifest(rows, man)
    print(f"manifest {man} n={len(rows)}")
    execute_manifest(rows)


if __name__ == "__main__":
    main()
