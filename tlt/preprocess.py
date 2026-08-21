"""Build temporally valid cached tables. Run before the training grid."""
from __future__ import annotations

from pathlib import Path

from tlt.datasets.oulad_temporal import load_oulad_bundle
from tlt.datasets.uci697_temporal import load_uci697_bundle
from tlt.protocol import freeze_protocol, load_protocol
from tlt.seeds import write_seed_plan

ROOT = Path(__file__).resolve().parents[1]


def verify_uci(bundle, checkpoint: str) -> None:
    forbidden = [n for n in bundle.feature_names if "2nd sem" in n]
    if forbidden:
        raise RuntimeError(f"UCI697 {checkpoint} leaked 2nd-sem features: {forbidden}")
    if checkpoint == "enrollment":
        bad = [n for n in bundle.feature_names if "1st sem" in n]
        if bad:
            raise RuntimeError(f"enrollment checkpoint leaked 1st-sem features: {bad}")


def verify_oulad(bundle, tau: int) -> None:
    forbidden = {"final_result", "date_unregistration", "label", "id_student"}
    hit = forbidden.intersection(bundle.feature_names)
    if hit:
        raise RuntimeError(f"OULAD day{tau} forbidden features: {hit}")


def main():
    proto = load_protocol()
    write_seed_plan()
    freeze_protocol(
        extra_paths=[
            ROOT / "tlt" / "datasets" / "oulad_temporal.py",
            ROOT / "tlt" / "datasets" / "uci697_temporal.py",
            ROOT / "tlt" / "metrics" / "queue.py",
            ROOT / "tlt" / "trainer.py",
            ROOT / "tlt" / "run_grid.py",
            ROOT / "artifacts" / "manifests" / "seed_plan.csv",
        ]
    )
    print("=== UCI697 semester1 ===")
    u1 = load_uci697_bundle("semester1", split_seed=proto["split_seed"])
    verify_uci(u1, "semester1")
    print(u1.metadata)
    print("features", u1.feature_names)
    print("=== UCI697 enrollment ===")
    u0 = load_uci697_bundle("enrollment", split_seed=proto["split_seed"])
    verify_uci(u0, "enrollment")
    print(u0.metadata)
    for tau in (28, 14, 56):
        print(f"=== OULAD day{tau} ===")
        b = load_oulad_bundle(tau=tau, split_seed=proto["split_seed"])
        verify_oulad(b, tau)
        print(b.metadata)
        print("n_features", len(b.feature_names), "features", b.feature_names)


if __name__ == "__main__":
    main()
