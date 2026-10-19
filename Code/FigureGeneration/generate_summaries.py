#!/usr/bin/env python3
"""Regenerate the numeric tables that back the quantitative figures."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from figure_utils import write_summaries
from io_utils import (
    IBM_RESULTS_DIR,
    SIMULATION_RESULTS_DIR,
    SUMMARY_OUTPUT_DIR,
    read_json,
    read_table,
)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=SIMULATION_RESULTS_DIR)
    parser.add_argument(
        "--ibm-json", type=Path, default=IBM_RESULTS_DIR / "ibm_qpu_results.json"
    )
    parser.add_argument("--output-dir", type=Path, default=SUMMARY_OUTPUT_DIR)
    args = parser.parse_args(argv)

    nonprivate = read_table(
        args.results_dir / "nonprivate_results.csv", {"dataset", "model", "auc"}
    )
    private = read_table(
        args.results_dir / "private_results.csv",
        {
            "dataset", "model", "seed", "eps_target", "realized_eps", "auc",
            "sigma", "sample_rate", "steps", "private_units",
        },
    )
    corrected = read_table(
        args.results_dir / "corrected_statistical_tests.csv", set()
    )
    ablation = read_table(
        args.results_dir / "ablation_results.csv",
        {"dataset", "n_features", "n_layers", "encoding", "auc"},
    )
    ibm = read_json(
        args.ibm_json,
        {"runtime", "circuit", "exact_statevector", "qpu", "agreement_to_exact"},
    )
    write_summaries(nonprivate, private, corrected, ablation, ibm, args.output_dir)
    print(f"Generated numeric figure data in {args.output_dir}")


if __name__ == "__main__":
    main()
