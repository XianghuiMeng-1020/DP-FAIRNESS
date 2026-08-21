"""Train one TLT run and freeze scores. Resume-safe."""
from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from tlt.datasets.oulad_temporal import load_oulad_bundle
from tlt.datasets.uci697_temporal import load_uci697_bundle
from tlt.protocol import load_protocol
from tlt.trainer import TLTTrainer

ROOT = Path(__file__).resolve().parents[1]


def load_bundle(dataset: str, checkpoint: str, split_seed: int):
    if dataset == "OULAD":
        tau = int(str(checkpoint).replace("day", ""))
        return load_oulad_bundle(tau=tau, split_seed=split_seed)
    if dataset == "UCI697":
        return load_uci697_bundle(checkpoint=checkpoint, split_seed=split_seed)
    raise ValueError(dataset)


def run_dir(run_id: str) -> Path:
    return ROOT / "artifacts" / "predictions" / run_id


def is_complete(path: Path) -> bool:
    needed = ["scores_test.npy", "y_test.npy", "ids_test.npy", "train_info.json", "config.json"]
    return path.exists() and all((path / n).exists() for n in needed)


def execute_run(entry: Dict[str, Any], overwrite: bool = False) -> Dict[str, Any]:
    proto = load_protocol()
    out = run_dir(entry["run_id"])
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(entry)
    cfg["protocol_version"] = proto.get("protocol_version")
    cfg["protocol_change_reason"] = proto.get("protocol_change_reason")
    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    if is_complete(out) and not overwrite:
        with open(out / "train_info.json", encoding="utf-8") as f:
            info = json.load(f)
        info["resumed"] = True
        return info

    started = datetime.now(timezone.utc).isoformat()
    try:
        bundle = load_bundle(entry["dataset"], entry["checkpoint"], entry["split_seed"])
        trainer = TLTTrainer(entry["architecture"], seed=entry["train_seed"])
        mech = entry["mechanism"]
        info = trainer.train(
            bundle.X_train,
            bundle.y_train,
            X_val=bundle.X_val,
            y_val=bundle.y_val,
            mechanism=mech,
            epsilon=entry.get("epsilon"),
            max_grad_norm=proto["privacy"]["clipping_norm"],
            epochs=proto["privacy"]["dp_epochs"] if mech == "DP-SGD" else proto["optimization"]["np_epochs"],
            batch_size=proto["privacy"]["dp_batch_size"] if mech == "DP-SGD" else proto["optimization"]["np_batch_size"],
            lr=proto["privacy"]["dp_lr"] if mech == "DP-SGD" else proto["optimization"]["np_lr"],
            weight_decay=float(entry.get("weight_decay", proto["optimization"]["weight_decay_default"])),
            early_stopping=proto["optimization"]["np_early_stopping"] if mech == "none" else False,
            early_stopping_patience=proto["optimization"]["np_early_stopping_patience"],
        )
        scores_test = trainer.predict_proba(bundle.X_test)[:, 1]
        scores_val = trainer.predict_proba(bundle.X_val)[:, 1]
        if not np.isfinite(scores_test).all():
            raise RuntimeError("nan_scores")
        if mech == "DP-SGD" and info.get("realized_epsilon") is None:
            raise RuntimeError("incomplete_privacy_accounting")

        np.save(out / "scores_test.npy", scores_test)
        np.save(out / "scores_val.npy", scores_val)
        np.save(out / "y_test.npy", bundle.y_test)
        np.save(out / "y_val.npy", bundle.y_val)
        np.save(out / "ids_test.npy", bundle.ids_test)
        np.save(out / "ids_val.npy", bundle.ids_val)
        if bundle.groups_test is not None:
            np.save(out / "groups_test.npy", bundle.groups_test)
        with open(out / "feature_names.json", "w", encoding="utf-8") as f:
            json.dump(bundle.feature_names, f, indent=2)
        info.update(
            {
                "run_id": entry["run_id"],
                "dataset": entry["dataset"],
                "checkpoint": entry["checkpoint"],
                "started_at_utc": started,
                "ended_at_utc": datetime.now(timezone.utc).isoformat(),
                "n_test": int(len(bundle.y_test)),
                "n_features": int(bundle.X_train.shape[1]),
                "status": "ok",
                "bundle_metadata": bundle.metadata,
            }
        )
        with open(out / "train_info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, default=str)
        return info
    except Exception as exc:
        payload = {
            "run_id": entry["run_id"],
            "status": "failed",
            "reason": str(exc),
            "traceback": traceback.format_exc(),
            "started_at_utc": started,
            "ended_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        with open(out / "failure.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return payload
