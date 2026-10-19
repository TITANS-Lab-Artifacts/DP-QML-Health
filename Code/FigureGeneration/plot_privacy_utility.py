#!/usr/bin/env python3
"""Generate the privacy-utility figure from project simulation tables."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from figure_utils import configure_style, privacy_utility_figure
from io_utils import FIGURE_OUTPUT_DIR, SIMULATION_RESULTS_DIR, read_table


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=SIMULATION_RESULTS_DIR,
        help="folder containing private_results.csv and nonprivate_results.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FIGURE_OUTPUT_DIR,
        help="destination for fig_privacy_utility.pdf and .png",
    )
    args = parser.parse_args(argv)

    # This figure needs the private curves and the non-private QNN references.
    private = read_table(
        args.results_dir / "private_results.csv",
        {"dataset", "model", "eps_target", "auc"},
    )
    nonprivate = read_table(
        args.results_dir / "nonprivate_results.csv",
        {"dataset", "model", "auc"},
    )
    configure_style()
    privacy_utility_figure(private, nonprivate, args.output_dir, args.output_dir)
    print(f"Generated {args.output_dir / 'fig_privacy_utility.pdf'}")


if __name__ == "__main__":
    main()
