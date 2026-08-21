"""Scientific (not publication-styled) TLT figure candidates."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "tlt_scientific"


def _load():
    metrics = ROOT / "artifacts" / "metrics"
    seed = pd.read_csv(metrics / "primary_seed_level_results.csv")
    ept = pd.read_csv(metrics / "excess_turnover_results.csv")
    loc_path = metrics / "cutoff_localization_results.csv"
    loc = pd.read_csv(loc_path) if loc_path.exists() else pd.DataFrame()
    return seed, ept, loc


def _mean_ci(part: pd.DataFrame, col: str):
    vals = part[col].dropna().values
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    m = vals.mean()
    if len(vals) == 1:
        return m, m, m
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return m, lo, hi


def figure_a(seed: pd.DataFrame):
    """Replacement vs capacity for NP retraining and DP epsilons."""
    OUT.mkdir(parents=True, exist_ok=True)
    for (ds, ck, arch), g in seed.groupby(["dataset", "checkpoint", "architecture"]):
        fig, ax = plt.subplots(figsize=(6, 4))
        series = {
            "ordinary_retraining": "NP retraining",
            "dp_vs_np_DP_eps10": "DP eps=10",
            "dp_vs_np_DP_eps5": "DP eps=5",
            "dp_vs_np_DP_eps1": "DP eps=1",
        }
        for comp, label in series.items():
            sub = g[g["comparison"] == comp]
            xs, ys, lo, hi = [], [], [], []
            for k, part in sub.groupby("k_frac"):
                m, l, h = _mean_ci(part, "turnover")
                xs.append(k)
                ys.append(m)
                lo.append(l)
                hi.append(h)
            if not xs:
                continue
            order = np.argsort(xs)
            xs = np.asarray(xs)[order]
            ys = np.asarray(ys)[order]
            lo = np.asarray(lo)[order]
            hi = np.asarray(hi)[order]
            ax.plot(xs, ys, marker="o", label=label)
            ax.fill_between(xs, lo, hi, alpha=0.15)
        ax.set_xlabel("capacity k")
        ax.set_ylabel("replacement fraction")
        ax.set_title(f"{ds} {ck} {arch}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / f"figA_turnover_{ds}_{ck}_{arch}.png", dpi=140)
        plt.close(fig)


def figure_b(ept: pd.DataFrame):
    OUT.mkdir(parents=True, exist_ok=True)
    if ept.empty:
        return
    for (ds, ck, arch), g in ept.groupby(["dataset", "checkpoint", "architecture"]):
        fig, ax = plt.subplots(figsize=(6, 4))
        for comp, part in g.groupby("comparison"):
            xs, ys, lo, hi = [], [], [], []
            for k, s in part.groupby("k_frac"):
                m, l, h = _mean_ci(s, "EPT")
                xs.append(k)
                ys.append(m)
                lo.append(l)
                hi.append(h)
            if not xs:
                continue
            order = np.argsort(xs)
            ax.plot(np.asarray(xs)[order], np.asarray(ys)[order], marker="o", label=comp)
            ax.fill_between(np.asarray(xs)[order], np.asarray(lo)[order], np.asarray(hi)[order], alpha=0.15)
        ax.axhline(0.0, color="black", lw=1)
        ax.set_xlabel("capacity k")
        ax.set_ylabel("EPT")
        ax.set_title(f"{ds} {ck} {arch}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / f"figB_ept_{ds}_{ck}_{arch}.png", dpi=140)
        plt.close(fig)


def figure_c(ept: pd.DataFrame):
    OUT.mkdir(parents=True, exist_ok=True)
    if ept.empty:
        return
    ept = ept.copy()
    ept["eps"] = ept["condition_alt"].str.replace("DP_eps", "", regex=False)
    for (ds, ck, arch), g in ept.groupby(["dataset", "checkpoint", "architecture"]):
        pivot = g.groupby(["eps", "k_frac"])["EPT"].mean().unstack()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        im = ax.imshow(pivot.values, aspect="auto", origin="lower")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("capacity")
        ax.set_ylabel("epsilon")
        ax.set_title(f"EPT surface {ds} {ck} {arch}")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(OUT / f"figC_surface_{ds}_{ck}_{arch}.png", dpi=140)
        plt.close(fig)
        pivot.to_csv(OUT / f"figC_surface_{ds}_{ck}_{arch}.csv")


def figure_d(seed: pd.DataFrame):
    OUT.mkdir(parents=True, exist_ok=True)
    dp = seed[seed["comparison"].astype(str).str.startswith("dp_vs_np_")]
    for (ds, ck, arch), g in dp.groupby(["dataset", "checkpoint", "architecture"]):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(g["turnover"], g["delta_outcome_capture"], alpha=0.5, s=18)
        ax.axhline(0.0, color="black", lw=1)
        ax.set_xlabel("queue replacement")
        ax.set_ylabel("delta Outcome-Capture@k")
        ax.set_title(f"{ds} {ck} {arch}")
        fig.tight_layout()
        fig.savefig(OUT / f"figD_turnover_vs_capture_{ds}_{ck}_{arch}.png", dpi=140)
        plt.close(fig)


def figure_e(loc: pd.DataFrame):
    OUT.mkdir(parents=True, exist_ok=True)
    if loc.empty:
        return
    for (ds, ck, arch, comp), g in loc.groupby(["dataset", "checkpoint", "architecture", "comparison"]):
        if "ordinary_retraining" not in str(comp) and "dp_vs_np_DP_eps5" not in str(comp):
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        for k, part in g.groupby("k_frac"):
            rates = part.groupby("bin")["transition_rate"].mean()
            ax.plot(range(len(rates)), rates.values, marker="o", label=f"k={k}")
            ax.set_xticks(range(len(rates)))
            ax.set_xticklabels(list(rates.index), rotation=45, ha="right")
        ax.set_ylabel("transition probability")
        ax.set_title(f"{ds} {ck} {arch} {comp}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / f"figE_cutoff_{ds}_{ck}_{arch}_{comp}.png", dpi=140)
        plt.close(fig)


def main():
    seed, ept, loc = _load()
    figure_a(seed)
    figure_b(ept)
    figure_c(ept)
    figure_d(seed)
    figure_e(loc)
    print(f"figures written to {OUT}")


if __name__ == "__main__":
    main()
