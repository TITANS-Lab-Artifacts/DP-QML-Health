"""Small I/O helpers shared by the result-figure scripts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SIMULATION_RESULTS_DIR = PROJECT_ROOT / "Results" / "Simulation" / "Tables"
IBM_RESULTS_DIR = PROJECT_ROOT / "Results" / "IBM_QPU"
FIGURE_OUTPUT_DIR = PROJECT_ROOT / "Results" / "Figures" / "Generated"
SUMMARY_OUTPUT_DIR = PROJECT_ROOT / "Results" / "Figures" / "Data"


def read_table(path: Path, required_columns: set[str]) -> pd.DataFrame:
    """Read one CSV file and report missing inputs or columns clearly."""

    if not path.is_file():
        raise FileNotFoundError(f"Required result file not found: {path}")
    table = pd.read_csv(path)
    missing = sorted(required_columns.difference(table.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    return table


def read_json(path: Path, required_keys: set[str]) -> dict:
    """Read one JSON file and validate its top-level fields."""

    if not path.is_file():
        raise FileNotFoundError(f"Required result file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    missing = sorted(required_keys.difference(payload))
    if missing:
        raise ValueError(f"{path} is missing keys: {', '.join(missing)}")
    return payload
