#!/usr/bin/env python3
"""Generate the clipping-sensitivity figure from private simulation results."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from figure_utils import clipping_figure, configure_style
from io_utils import FIGURE_OUTPUT_DIR, SIMULATION_RESULTS_DIR, read_table


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-file",
        type=Path,
        default=SIMULATION_RESULTS_DIR / "private_results.csv",
        help="private-results CSV used for clipping and noise comparisons",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FIGURE_OUTPUT_DIR,
        help="destination for fig_clipping.pdf and .png",
    )
    args = parser.parse_args(argv)

    private = read_table(
        args.input_file,
        {"dataset", "model", "seed", "auc", "eps_target", "sigma"},
    )
    configure_style()
    clipping_figure(private, args.output_dir, args.output_dir)
    print(f"Generated {args.output_dir / 'fig_clipping.pdf'}")


if __name__ == "__main__":
    main()
