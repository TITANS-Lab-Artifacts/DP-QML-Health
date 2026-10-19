#!/usr/bin/env python3
"""Build a complete, checksummed inventory of the Unit-Aware DP-QML Health project."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "Documentation" / "FILE_MANIFEST.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe(relative: Path) -> tuple[str, str]:
    """Return a stable category and concise pattern-aware description."""

    path = relative.as_posix()
    name = relative.name
    suffix = relative.suffix.lower()

    exact = {
        "README.md": "Main organization, setup, execution, and interpretation guide.",
        ".gitignore": "Generated/cache/demo file exclusions.",
        "Code/Simulation/config.json": "Complete simulation parameter configuration.",
        "Code/Simulation/prepare_data.py": "Verified download and dataset preparation stage.",
        "Code/Simulation/qpriv_utils.py": "Shared paths, preprocessing, models, training, DP, metrics, and I/O.",
        "Code/Simulation/run_nonprivate.py": "Non-private model sweep.",
        "Code/Simulation/run_private.py": "Private model and epsilon sweep.",
        "Code/Simulation/run_ablation.py": "Private QNN architecture ablation sweep.",
        "Code/Simulation/analyze_results.py": "Summary and corrected statistical analysis.",
        "Code/Simulation/run_all.py": "Complete ordered simulation workflow.",
        "Code/IBM_QPU/run_ibm_qpu.py": "Exact validation and optional IBM Runtime execution.",
        "Code/FigureGeneration/plot_privacy_utility.py": "Privacy-utility figure entry point.",
        "Code/FigureGeneration/plot_clipping.py": "Clipping-sensitivity figure entry point.",
        "Code/FigureGeneration/plot_ablation.py": "Architecture-ablation figure entry point.",
        "Code/FigureGeneration/plot_ibm_validation.py": "IBM-validation figure entry point.",
        "Code/FigureGeneration/generate_all_figures.py": "Convenience runner for all result figures.",
        "Code/FigureGeneration/generate_summaries.py": "Numeric figure-summary generator.",
        "Code/FigureGeneration/figure_utils.py": "Shared quantitative plotting implementation and style.",
        "Code/FigureGeneration/io_utils.py": "Canonical figure paths and input validation.",
        "Code/build_manifest.py": "Complete checksummed file-inventory generator.",
        "Code/validate_project.py": "Project structure and result-integrity validator.",
    }
    if path in exact:
        return "code" if path.startswith("Code/") else "root", exact[path]

    if path.startswith("Code/"):
        if name == "README.md":
            return "documentation", "Subsystem usage guide."
        if name == "requirements.txt":
            return "dependency", "Subsystem Python dependency constraints."
        return "code", "Project source or configuration file."
    if path.startswith("Data/Simulation/raw/"):
        return "input-data", "Verified raw source dataset."
    if path.startswith("Data/Simulation/"):
        if name == "dataset_manifest.json":
            return "input-data", "Dataset source and SHA-256 provenance manifest."
        return "input-data", "Prepared simulation input dataset."
    if path.startswith("Results/Simulation/Artifacts/nonprivate/"):
        return "model-artifact", "Non-private model checkpoint for the dataset/model/seed encoded in the path."
    if path.startswith("Results/Simulation/Artifacts/private_ablation/"):
        return "model-artifact", "Private QNN ablation checkpoint; feature/layer/encoding/seed are encoded in the name."
    if path.startswith("Results/Simulation/Artifacts/private/"):
        return "model-artifact", "Private model checkpoint; dataset/model/epsilon/seed are encoded in the path."
    if path.startswith("Results/Simulation/Artifacts/"):
        return "preprocessor-artifact", "Public-only fitted preprocessor for the dataset/seed encoded in the name."
    if path.startswith("Results/Simulation/Tables/"):
        form = "CSV table" if suffix == ".csv" else "strict JSON records"
        return "simulation-result", f"Archived simulation {form}: {relative.stem.replace('_', ' ')}."
    if path.startswith("Results/IBM_QPU/"):
        if "calibration" in name:
            return "ibm-result", "Timestamped IBM backend calibration snapshot."
        return "ibm-result", f"Archived IBM record: {relative.stem.replace('_', ' ')}."
    if path.startswith("Results/Figures/Generated/"):
        form = "vector PDF" if suffix == ".pdf" else "400-DPI PNG"
        return "figure-result", f"Quantitative result figure ({form}): {relative.stem}."
    if path.startswith("Results/Figures/Data/"):
        return "figure-data", f"Numeric backing data: {relative.stem.replace('_', ' ')}."
    if path.startswith("Documentation/"):
        return "documentation", f"Project documentation: {relative.stem.replace('_', ' ').title()}."
    return "other", "Delivered project file."


def main() -> None:
    files = sorted(
        path
        for path in PROJECT_ROOT.rglob("*")
        if path.is_file()
        and path != OUTPUT_PATH
        and "__pycache__" not in path.parts
        and path.name != ".DS_Store"
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "category", "description", "bytes", "sha256"],
        )
        writer.writeheader()
        for path in files:
            relative = path.relative_to(PROJECT_ROOT)
            category, description = describe(relative)
            writer.writerow(
                {
                    "path": relative.as_posix(),
                    "category": category,
                    "description": description,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
        # A manifest cannot contain its own stable byte count/hash: adding that
        # value would change the file again.  Include the file with explicit
        # self-reference markers so every delivered path still has one row.
        writer.writerow(
            {
                "path": "Documentation/FILE_MANIFEST.csv",
                "category": "documentation",
                "description": "This complete file inventory (self-referential digest intentionally omitted).",
                "bytes": "self",
                "sha256": "self",
            }
        )
    print(f"Wrote {len(files) + 1} entries to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
