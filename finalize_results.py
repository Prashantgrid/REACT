#!/usr/bin/env python3
"""Regenerate affected figures and record the exact input/code fingerprints."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("results", type=Path)
parser.add_argument("--storage-file", required=True, type=Path)
parser.add_argument("--snapshot-file", required=True, type=Path)
args = parser.parse_args()
code = Path(__file__).resolve().parent
project = code if (code / "figure_sources").exists() else code.parent
figures = project / "figure_sources"
for path in (args.results / "figure_data").glob("*.json"):
    shutil.copy2(path, figures / "data" / path.name)
shutil.copy2(args.results / "revision" / "robustness_matrix.csv", figures / "data" / "robustness_matrix.csv")
for name in ("fig_rankings.py", "fig_relative_weakness_map.py", "fig_robustness.py"):
    subprocess.run([sys.executable, str(figures / name)], cwd=figures, check=True)
destination = project / "manuscript" / "images"
if destination.exists():
    for name in ("fig03_M0_M1_M2.pdf", "fig04_baseline.pdf", "fig05_relative_weakness_map.pdf", "fig06_robustness.pdf"):
        shutil.copy2(figures / "out" / name, destination / name)
    subprocess.run([sys.executable, str(code / "update_manuscript_tables.py"),
                    str(args.results), str(project / "manuscript" / "REACT_ESR.tex")], check=True)

def fingerprint(path):
    return {"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

def snapshot_date(path):
    with path.open(newline="") as handle:
        dates = {row.get("snapshot_date") for row in csv.DictReader(handle)}
    if len(dates) != 1:
        raise ValueError(f"Storage input must describe one snapshot: {path}")
    return dates.pop()

manifest = {"main_storage": fingerprint(args.storage_file),
            "later_snapshot": fingerprint(args.snapshot_file),
            "main_snapshot_date": snapshot_date(args.storage_file),
            "sensitivity_snapshot_date": snapshot_date(args.snapshot_file),
            "storage_column": "battery_energy_MWh_excluding_duration_flags",
            "hazard_iterations": 2000, "profile_draws": 10000,
            "permutation_replicates": 5000, "seed": 12345,
            "full_model_bootstrap_samples": 40,
            "source_files": [fingerprint(path) for path in sorted(code.glob("*.py"))]}
(args.results / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
