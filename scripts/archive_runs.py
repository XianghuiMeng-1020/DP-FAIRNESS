#!/usr/bin/env python3
"""Archive old/contaminated runs without deletion"""
import json
import shutil
import sys
from pathlib import Path
from typing import List, Dict, Any

sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

base_dir = Path(__file__).parent.parent
runs_dir = base_dir / "outputs" / "runs"
archive_dir = runs_dir / "_archive_old_negctrl"

def archive_runs(pattern: str = None, run_ids: List[str] = None) -> Dict[str, Any]:
    """Archive runs matching pattern or run_ids"""
    archived = []
    failed = []
    
    # Create archive directory
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    # Load plan to identify runs
    plan_path = base_dir / "outputs" / "reports" / "experiment_plan_fast.json"
    if not plan_path.exists():
        return {"error": "Plan file not found"}
    
    with open(plan_path, 'r', encoding='utf-8') as f:
        plan = json.load(f)
    
    # Identify runs to archive
    runs_to_archive = []
    
    if run_ids:
        # Archive specific run_ids
        for entry in plan:
            if entry.get("run_id") in run_ids:
                runs_to_archive.append(entry)
    elif pattern:
        # Archive by pattern
        for entry in plan:
            run_id = entry.get("run_id", "")
            if pattern.lower() in run_id.lower():
                runs_to_archive.append(entry)
    else:
        # Default: archive old negative control runs
        for entry in plan:
            run_id = entry.get("run_id", "")
            if "negative_control" in run_id.lower():
                runs_to_archive.append(entry)
    
    print(f"Found {len(runs_to_archive)} runs to archive")
    
    # Archive each run
    for entry in runs_to_archive:
        run_id = entry.get("run_id")
        source_dir = runs_dir / run_id
        
        if not source_dir.exists():
            print(f"  WARNING: {run_id} directory not found, skipping")
            continue
        
        try:
            dest_dir = archive_dir / run_id
            if dest_dir.exists():
                print(f"  WARNING: {run_id} already archived, skipping")
                continue
            
            # Move directory
            shutil.move(str(source_dir), str(dest_dir))
            archived.append(run_id)
            print(f"  Archived: {run_id}")
        except Exception as e:
            failed.append({"run_id": run_id, "error": str(e)})
            print(f"  ERROR archiving {run_id}: {e}")
    
    # Save archive manifest
    manifest = {
        "archived_runs": archived,
        "failed": failed,
        "total_archived": len(archived),
        "total_failed": len(failed)
    }
    
    manifest_path = base_dir / "paper" / "archived_runs.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    print(f"\nArchive complete:")
    print(f"  Archived: {len(archived)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Manifest saved to: {manifest_path}")
    
    return manifest

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", help="Pattern to match run_ids")
    parser.add_argument("--run-ids", nargs="+", help="Specific run_ids to archive")
    parser.add_argument("--negative-controls", action="store_true", help="Archive all negative control runs")
    
    args = parser.parse_args()
    
    if args.run_ids:
        result = archive_runs(run_ids=args.run_ids)
    elif args.pattern:
        result = archive_runs(pattern=args.pattern)
    elif args.negative_controls:
        result = archive_runs(pattern="negative_control")
    else:
        # Default: archive old negative controls
        print("Archiving old negative control runs...")
        result = archive_runs(pattern="negative_control")
