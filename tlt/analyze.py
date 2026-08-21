"""Build all TLT scientific result surfaces from frozen scores."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from tlt.metrics.calibration import expected_calibration_error
from tlt.metrics.bootstrap import percentile_bootstrap_ci
from tlt.metrics.cutoff import localize_transitions, summarize_localization
from tlt.metrics.queue import (
    build_priority_queue,
    excess_prioritization_turnover,
    jaccard_index,
    queue_utility,
    turnover_fraction,
)
from tlt.protocol import load_protocol
from tlt.run_one import run_dir

ROOT = Path(__file__).resolve().parents[1]
CAPS = [0.05, 0.10, 0.20, 0.30]


def _load_run(run_id: str) -> Dict:
    d = run_dir(run_id)
    with open(d / "train_info.json", encoding="utf-8") as f:
        info = json.load(f)
    with open(d / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    return {
        "info": info,
        "cfg": cfg,
        "scores": np.load(d / "scores_test.npy"),
        "y": np.load(d / "y_test.npy"),
        "ids": np.load(d / "ids_test.npy", allow_pickle=True),
        "scores_val": np.load(d / "scores_val.npy") if (d / "scores_val.npy").exists() else None,
        "y_val": np.load(d / "y_val.npy") if (d / "y_val.npy").exists() else None,
        "groups": np.load(d / "groups_test.npy") if (d / "groups_test.npy").exists() else None,
    }


def _index_manifest(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _ok_rows(man: pd.DataFrame) -> pd.DataFrame:
    keep = []
    for _, r in man.iterrows():
        d = run_dir(r["run_id"])
        if (d / "train_info.json").exists() and (d / "scores_test.npy").exists():
            with open(d / "train_info.json", encoding="utf-8") as f:
                info = json.load(f)
            if info.get("status") == "ok":
                keep.append(r)
    return pd.DataFrame(keep)


def _key(r) -> Tuple:
    return (r["dataset"], r["checkpoint"], r["architecture"], int(r["replicate"]))


def analyze(manifest_path: Path | None = None) -> None:
    proto = load_protocol()
    paths = []
    if manifest_path is not None:
        paths = [manifest_path]
    else:
        paths = [ROOT / "artifacts" / "manifests" / "experiment_manifest.csv"]
        sec = ROOT / "artifacts" / "manifests" / "experiment_manifest_secondary.csv"
        if sec.exists():
            paths.append(sec)
    man = pd.concat([_index_manifest(p) for p in paths], ignore_index=True)
    man = _ok_rows(man)
    by = defaultdict(dict)
    for _, r in man.iterrows():
        by[_key(r)][r["condition"]] = r["run_id"]

    seed_rows = []
    util_rows = []
    loc_rows = []
    acct_rows = []
    group_rows = []
    calib_rows = []

    for key, conds in by.items():
        dataset, checkpoint, arch, rep = key
        if "NP_A" not in conds:
            continue
        ref = _load_run(conds["NP_A"])
        y = ref["y"]
        ids = ref["ids"]
        scores_a = ref["scores"]

        # accounting / calibration for every available condition
        for cond, rid in conds.items():
            run = _load_run(rid)
            info = run["info"]
            acct_rows.append(
                {
                    "dataset": dataset,
                    "checkpoint": checkpoint,
                    "architecture": arch,
                    "condition": cond,
                    "replicate": rep,
                    "run_id": rid,
                    "target_epsilon": info.get("target_epsilon"),
                    "realized_epsilon": info.get("realized_epsilon"),
                    "delta": info.get("delta"),
                    "clipping_norm": info.get("clipping_norm"),
                    "noise_multiplier": info.get("noise_multiplier"),
                    "sample_rate": info.get("sample_rate"),
                    "optimizer_steps": info.get("optimizer_steps"),
                    "epoch_count": info.get("epoch_count"),
                    "early_stopped": info.get("early_stopped"),
                    "train_auc": info.get("train_auc"),
                    "val_auc": info.get("val_auc"),
                    "test_auc": float(roc_auc_score(run["y"], run["scores"]))
                    if len(np.unique(run["y"])) > 1
                    else np.nan,
                }
            )
            calib_rows.append(
                {
                    "dataset": dataset,
                    "checkpoint": checkpoint,
                    "architecture": arch,
                    "condition": cond,
                    "replicate": rep,
                    "ece": expected_calibration_error(run["y"], run["scores"]),
                    "brier": float(brier_score_loss(run["y"], run["scores"])),
                    "log_loss": float(
                        log_loss(run["y"], np.clip(run["scores"], 1e-6, 1 - 1e-6), labels=[0, 1])
                    ),
                }
            )

        spec_cond = _select_spec_condition(conds, by_val=True)
        matched_map = _val_matched_conditions(conds, ref)

        for k in proto["capacities"]:
            qa = build_priority_queue(scores_a, ids, k)
            util_a = queue_utility(y, qa)

            def pair(cond_name: str, comparison: str):
                if cond_name not in conds:
                    return
                alt = _load_run(conds[cond_name])
                qb = build_priority_queue(alt["scores"], ids, k)
                util_b = queue_utility(y, qb)
                t = turnover_fraction(qa, qb)
                j = jaccard_index(qa, qb)
                seed_rows.append(
                    {
                        "dataset": dataset,
                        "checkpoint": checkpoint,
                        "architecture": arch,
                        "replicate": rep,
                        "k_frac": k,
                        "comparison": comparison,
                        "condition_ref": "NP_A",
                        "condition_alt": cond_name,
                        "turnover": t,
                        "jaccard": j,
                        "precision_ref": util_a["precision_at_k"],
                        "precision_alt": util_b["precision_at_k"],
                        "delta_precision": util_b["precision_at_k"] - util_a["precision_at_k"],
                        "outcome_capture_ref": util_a["outcome_capture_at_k"],
                        "outcome_capture_alt": util_b["outcome_capture_at_k"],
                        "delta_outcome_capture": util_b["outcome_capture_at_k"]
                        - util_a["outcome_capture_at_k"],
                    }
                )
                loc = localize_transitions(scores_a, alt["scores"], ids, y, k)
                loc_sum = summarize_localization(loc)
                loc_sum["dataset"] = dataset
                loc_sum["checkpoint"] = checkpoint
                loc_sum["architecture"] = arch
                loc_sum["replicate"] = rep
                loc_sum["k_frac"] = k
                loc_sum["comparison"] = comparison
                loc_rows.append(loc_sum)

                if ref["groups"] is not None:
                    for g in np.unique(ref["groups"]):
                        mask = ref["groups"] == g
                        if mask.sum() < proto["minimum_cell_rule"]["n_total"]:
                            continue
                        y_g = y[mask]
                        if (y_g == 1).sum() < proto["minimum_cell_rule"]["n_positive"]:
                            continue
                        if (y_g == 0).sum() < proto["minimum_cell_rule"]["n_negative"]:
                            continue
                        ug_a = queue_utility(y_g, qa[mask])
                        ug_b = queue_utility(y_g, qb[mask])
                        group_rows.append(
                            {
                                "dataset": dataset,
                                "checkpoint": checkpoint,
                                "architecture": arch,
                                "replicate": rep,
                                "k_frac": k,
                                "comparison": comparison,
                                "group": int(g),
                                "precision_ref": ug_a["precision_at_k"],
                                "precision_alt": ug_b["precision_at_k"],
                                "outcome_capture_ref": ug_a["outcome_capture_at_k"],
                                "outcome_capture_alt": ug_b["outcome_capture_at_k"],
                                "queue_share_ref": float(qa[mask].sum() / qa.sum()) if qa.sum() else np.nan,
                                "queue_share_alt": float(qb[mask].sum() / qb.sum()) if qb.sum() else np.nan,
                            }
                        )

            if "NP_B" in conds:
                pair("NP_B", "ordinary_retraining")
            for cond in conds:
                if str(cond).startswith("DP_eps"):
                    pair(cond, f"dp_vs_np_{cond}")
            if spec_cond:
                pair(spec_cond, "specification_baseline")
            for cond, matched in matched_map.items():
                if matched:
                    pair(matched, f"val_auc_matched_{cond}")

    seed_df = pd.DataFrame(seed_rows)
    out_metrics = ROOT / "artifacts" / "metrics"
    out_stats = ROOT / "artifacts" / "statistics"
    out_metrics.mkdir(parents=True, exist_ok=True)
    out_stats.mkdir(parents=True, exist_ok=True)
    seed_df.to_csv(out_metrics / "primary_seed_level_results.csv", index=False)

    # EPT: for each DP comparison, subtract same-replicate ordinary retraining
    ept_rows = []
    if not seed_df.empty:
        for keys, part in seed_df.groupby(
            ["dataset", "checkpoint", "architecture", "replicate", "k_frac"]
        ):
            t_np = part.loc[part["comparison"] == "ordinary_retraining", "turnover"]
            if t_np.empty:
                continue
            t_np_v = float(t_np.iloc[0])
            for _, row in part.iterrows():
                if not str(row["comparison"]).startswith("dp_vs_np_"):
                    continue
                ept_rows.append(
                    {
                        "dataset": keys[0],
                        "checkpoint": keys[1],
                        "architecture": keys[2],
                        "replicate": keys[3],
                        "k_frac": keys[4],
                        "comparison": row["comparison"],
                        "condition_alt": row["condition_alt"],
                        "T_NP": t_np_v,
                        "T_DP": row["turnover"],
                        "EPT": float(row["turnover"] - t_np_v),
                        "J_NP": float(
                            part.loc[part["comparison"] == "ordinary_retraining", "jaccard"].iloc[0]
                        ),
                        "J_DP": row["jaccard"],
                    }
                )
    ept_df = pd.DataFrame(ept_rows)
    ept_df.to_csv(out_metrics / "excess_turnover_results.csv", index=False)

    def agg_table(df: pd.DataFrame, value_col: str, group_cols: List[str]) -> pd.DataFrame:
        recs = []
        if df.empty:
            return pd.DataFrame()
        for g, part in df.groupby(group_cols):
            stats = percentile_bootstrap_ci(part[value_col].values, n_boot=proto["n_bootstrap"])
            rec = dict(zip(group_cols, g if isinstance(g, tuple) else (g,)))
            rec["metric"] = value_col
            rec.update(stats)
            if value_col == "turnover" and "comparison" in part.columns:
                # nothing extra
                pass
            recs.append(rec)
        return pd.DataFrame(recs)

    ordinary = seed_df[seed_df["comparison"] == "ordinary_retraining"].copy() if not seed_df.empty else seed_df
    ordinary_agg = agg_table(
        ordinary, "turnover", ["dataset", "checkpoint", "architecture", "k_frac", "comparison"]
    )
    ordinary.to_csv(out_metrics / "ordinary_stability_results.csv", index=False)
    if not ept_df.empty:
        ept_agg = agg_table(
            ept_df, "EPT", ["dataset", "checkpoint", "architecture", "k_frac", "comparison"]
        )
        extra = []
        for g, part in ept_df.groupby(["dataset", "checkpoint", "architecture", "k_frac", "comparison"]):
            extra.append(
                {
                    "dataset": g[0],
                    "checkpoint": g[1],
                    "architecture": g[2],
                    "k_frac": g[3],
                    "comparison": g[4],
                    "prop_T_DP_gt_T_NP": float((part["T_DP"] > part["T_NP"]).mean()),
                }
            )
        ept_agg = ept_agg.merge(pd.DataFrame(extra), how="left")
        ept_agg.to_csv(out_stats / "excess_turnover_aggregate.csv", index=False)

    dp = seed_df[seed_df["comparison"].astype(str).str.startswith("dp_vs_np_")] if not seed_df.empty else seed_df
    dp_agg = agg_table(dp, "turnover", ["dataset", "checkpoint", "architecture", "k_frac", "comparison"])
    spec = seed_df[seed_df["comparison"] == "specification_baseline"] if not seed_df.empty else seed_df
    spec_agg = agg_table(spec, "turnover", ["dataset", "checkpoint", "architecture", "k_frac", "comparison"])

    primary_agg = pd.concat([ordinary_agg, dp_agg, spec_agg], ignore_index=True)
    primary_agg.to_csv(out_metrics / "primary_aggregate_results.csv", index=False)
    seed_df.to_csv(out_metrics / "queue_utility_results.csv", index=False)

    if loc_rows:
        loc_all = pd.concat(loc_rows, ignore_index=True)
        loc_all.to_csv(out_metrics / "cutoff_localization_results.csv", index=False)
    acct_dir = ROOT / "artifacts" / "dp_accounting"
    acct_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(acct_rows).to_csv(acct_dir / "privacy_accounting_results.csv", index=False)
    pd.DataFrame(calib_rows).to_csv(out_metrics / "calibration_secondary.csv", index=False)
    if group_rows:
        pd.DataFrame(group_rows).to_csv(out_metrics / "group_diagnostics_secondary.csv", index=False)

    # architecture / checkpoint robustness slices
    if not ept_df.empty:
        ept_df.to_csv(out_metrics / "architecture_robustness_results.csv", index=False)
        ept_df.to_csv(out_metrics / "checkpoint_robustness_results.csv", index=False)

    print(f"wrote seed-level {len(seed_df)} rows; EPT {len(ept_df)} rows")


def _select_spec_condition(conds: Dict[str, str], by_val: bool = True) -> str | None:
    grid = [c for c in conds if str(c).startswith("NP_WD_")]
    if not grid:
        return None
    # pick the grid member with highest mean-over-nothing: use this replicate's val AUC
    best = None
    best_auc = -np.inf
    for c in grid:
        run = _load_run(conds[c])
        auc = run["info"].get("val_auc")
        if auc is not None and auc > best_auc:
            best_auc = auc
            best = c
    return best


def _val_matched_conditions(conds: Dict[str, str], ref: Dict) -> Dict[str, str]:
    pool = ["NP_A"] + [c for c in conds if str(c).startswith("NP_WD_")]
    pool_auc = {}
    for c in pool:
        if c not in conds and c != "NP_A":
            continue
        rid = conds.get(c, conds.get("NP_A"))
        if c == "NP_A":
            rid = conds["NP_A"]
        run = _load_run(rid)
        pool_auc[c] = run["info"].get("val_auc")
    matched = {}
    for c in conds:
        if not str(c).startswith("DP_eps"):
            continue
        dp_auc = _load_run(conds[c])["info"].get("val_auc")
        if dp_auc is None:
            continue
        best = min(pool_auc, key=lambda name: abs((pool_auc[name] or 0) - dp_auc))
        matched[c] = best
    return matched


if __name__ == "__main__":
    analyze()
