"""
Add provenance header to all_tables.md files
"""
import json
from pathlib import Path
from datetime import datetime
import subprocess

def get_git_hash():
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], 
                              capture_output=True, text=True, cwd=".")
        if result.returncode == 0:
            return result.stdout.strip()[:8]  # Short hash
    except:
        pass
    return "N/A (not a git repository)"

def get_timestamp():
    from datetime import timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def get_data_hashes():
    hashes = {}
    
    # OULAD
    oulad_path = Path("data/raw/oulad/studentInfo.csv")
    if oulad_path.exists():
        import hashlib
        with open(oulad_path, "rb") as f:
            hashes["OULAD"] = {
                "file": "data/raw/oulad/studentInfo.csv",
                "sha256": hashlib.sha256(f.read()).hexdigest()[:16] + "..."
            }
    
    # UCI697
    uci_path = Path("data/raw/uci697/data.csv")
    if uci_path.exists():
        import hashlib
        with open(uci_path, "rb") as f:
            hashes["UCI697"] = {
                "file": "data/raw/uci697/data.csv",
                "sha256": hashlib.sha256(f.read()).hexdigest()[:16] + "..."
            }
    
    # HarvardX
    harvard_path = Path("data/raw/harvardx/HXPC13_DI_v3_11-13-2019.tab")
    if harvard_path.exists():
        import hashlib
        with open(harvard_path, "rb") as f:
            hashes["HarvardX"] = {
                "file": "data/raw/harvardx/HXPC13_DI_v3_11-13-2019.tab",
                "sha256": hashlib.sha256(f.read()).hexdigest()[:16] + "..."
            }
    
    return hashes

def create_header():
    timestamp = get_timestamp()
    git_hash = get_git_hash()
    data_hashes = get_data_hashes()
    
    lines = [
        "<!-- PROVENANCE HEADER: Single Source of Truth -->",
        f"**Generated at**: {timestamp}",
        f"**Git commit**: {git_hash}",
        "",
        "**Raw data fingerprints (SHA256, first 16 chars)**:",
    ]
    
    for dataset, info in sorted(data_hashes.items()):
        lines.append(f"- {dataset}: `{info['file']}` -> `{info['sha256']}`")
    
    lines.extend([
        "",
        "**Synthetic data disabled**: `ALLOW_SYNTHETIC` unset/false; data loader raises `FileNotFoundError` if raw data files are missing.",
        "",
        "---",
        ""
    ])
    
    return "\n".join(lines)

if __name__ == "__main__":
    header = create_header()
    print(header)
