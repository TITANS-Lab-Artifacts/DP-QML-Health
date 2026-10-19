#!/usr/bin/env python3
"""Generate the QNN architecture-ablation figure from project results."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from figure_utils import ablation_figure, configure_style
from io_utils import FIGURE_OUTPUT_DIR, SIMULATION_RESULTS_DIR, read_table


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-file",
        type=Path,
        default=SIMULATION_RESULTS_DIR / "ablation_results.csv",
        help="architecture-ablation result CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FIGURE_OUTPUT_DIR,
        help="destination for fig_ablation.pdf and .png",
    )
    args = parser.parse_args(argv)

    ablation = read_table(
        args.input_file,
        {"dataset", "n_features", "n_layers", "encoding", "auc"},
    )
    configure_style()
    ablation_figure(ablation, args.output_dir, args.output_dir)
    print(f"Generated {args.output_dir / 'fig_ablation.pdf'}")


if __name__ == "__main__":
    main()
