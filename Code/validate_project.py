#!/usr/bin/env python3
"""Validate project structure, result records, paths, and figure scope."""

from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES = PROJECT_ROOT / "Results" / "Simulation" / "Tables"

EXPECTED_ROWS = {
    "dataset_info": 4,
    "nonprivate_results": 100,
    "private_lr_selection": 20,
    "private_results": 600,
    "ablation_results": 330,
    "ablation_skipped": 30,
    "model_summary": 140,
    "corrected_statistical_tests": 24,
    "privacy_accounting": 600,
}

REQUIRED_PATHS = [
    "Code/Simulation/config.json",
    "Code/Simulation/run_all.py",
    "Code/IBM_QPU/run_ibm_qpu.py",
    "Code/FigureGeneration/plot_privacy_utility.py",
    "Code/FigureGeneration/plot_clipping.py",
    "Code/FigureGeneration/plot_ablation.py",
    "Code/FigureGeneration/plot_ibm_validation.py",
    "Results/IBM_QPU/ibm_qpu_results.json",
]

REMOVED_DIAGRAM_MARKERS = {
    "diagram_spec.py",
    "generate_diagram_figures.py",
    "export_drawio_sources.py",
    "fig_background.pdf",
    "fig_background.png",
    "fig_system_architecture.pdf",
    "fig_system_architecture.png",
    "fig_quantum_circuit.pdf",
    "fig_quantum_circuit.png",
}


def csv_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    errors: list[str] = []

    for relative in REQUIRED_PATHS:
        if not (PROJECT_ROOT / relative).is_file():
            errors.append(f"missing required file: {relative}")

    delivered_names = {path.name for path in PROJECT_ROOT.rglob("*") if path.is_file()}
    unexpected = sorted(REMOVED_DIAGRAM_MARKERS.intersection(delivered_names))
    if unexpected:
        errors.append(f"diagram-only files remain: {', '.join(unexpected)}")
    if any(path.suffix == ".drawio" for path in PROJECT_ROOT.rglob("*")):
        errors.append("Draw.io source remains in the project package")

    for stem, expected in EXPECTED_ROWS.items():
        csv_path = TABLES / f"{stem}.csv"
        json_path = TABLES / f"{stem}.json"
        if not csv_path.is_file() or not json_path.is_file():
            errors.append(f"missing CSV/JSON pair: {stem}")
            continue
        rows = csv_rows(csv_path)
        with json_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if len(rows) != expected:
            errors.append(f"{csv_path.name}: expected {expected} rows, found {len(rows)}")
        if not isinstance(payload, list) or len(payload) != expected:
            actual = len(payload) if isinstance(payload, list) else type(payload).__name__
            errors.append(f"{json_path.name}: expected {expected} records, found {actual}")

    environment_path = TABLES / "environment.json"
    try:
        with environment_path.open(encoding="utf-8") as handle:
            environment = json.load(handle)
        if not isinstance(environment, dict):
            errors.append("environment.json must contain an object")
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"invalid environment.json: {exc}")

    artifact_files = [
        path
        for path in (PROJECT_ROOT / "Results" / "Simulation" / "Artifacts").rglob("*")
        if path.is_file()
    ]
    if len(artifact_files) != 1010:
        errors.append(f"expected 1,010 simulation artifacts, found {len(artifact_files)}")

    for table_name, field in (
        ("nonprivate_results.csv", "model_artifact"),
        ("private_results.csv", "model_artifact"),
        ("ablation_results.csv", "model_artifact"),
        ("private_lr_selection.csv", "preprocessor_artifact"),
    ):
        for row_number, row in enumerate(csv_rows(TABLES / table_name), start=2):
            value = row.get(field, "")
            if value and not (PROJECT_ROOT / value).is_file():
                errors.append(f"{table_name}:{row_number}: missing artifact {value}")

    figure_files = list((PROJECT_ROOT / "Results" / "Figures" / "Generated").glob("fig_*.*"))
    if len(figure_files) != 8:
        errors.append(f"expected 8 generated figure files, found {len(figure_files)}")

    for path in PROJECT_ROOT.rglob("*.json"):
        try:
            with path.open(encoding="utf-8") as handle:
                json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid JSON {path.relative_to(PROJECT_ROOT)}: {exc}")

    if errors:
        print("Validation failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("Validation passed")
    print("- canonical directory structure present")
    print("- 1,030 completed fit records and 30 skip records verified")
    print("- 1,010 artifact files and every referenced artifact path verified")
    print("- all JSON files parsed successfully")
    print("- four PDF/PNG result-figure pairs present")
    print("- diagram-only files absent")


if __name__ == "__main__":
    main()
