"""Load and freeze the TLT protocol."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs" / "tlt" / "tlt_protocol.yaml"


def load_protocol(path: Path = PROTOCOL_PATH) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def freeze_protocol(
    extra_paths: List[Path] | None = None,
    out_path: Path | None = None,
) -> Dict[str, Any]:
    proto = load_protocol()
    paths = [PROTOCOL_PATH]
    if extra_paths:
        paths.extend(extra_paths)
    hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): file_sha256(p) for p in paths if p.exists()}
    payload = {
        "protocol_id": proto["protocol_id"],
        "protocol_version": proto["protocol_version"],
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_declared_frozen_utc": proto.get("frozen_utc"),
        "file_hashes_sha256": hashes,
        "protocol": proto,
    }
    out = out_path or (ROOT / "artifacts" / "manifests" / "TLT_PROTOCOL_FREEZE.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def train_seed(role: str, replicate: int) -> int:
    if role == "NP_A":
        return 20000 + replicate
    if role == "NP_B":
        return 30000 + replicate
    if role in {"DP", "NP_SPEC", "NP_GRID"}:
        return 20000 + replicate
    raise ValueError(role)
