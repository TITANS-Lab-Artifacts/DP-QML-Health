#!/usr/bin/env python3
"""Generate the IBM QPU validation figure from the completed hardware record."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from figure_utils import configure_style, ibm_validation_figure
from io_utils import FIGURE_OUTPUT_DIR, IBM_RESULTS_DIR, read_json


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-file",
        type=Path,
        default=IBM_RESULTS_DIR / "ibm_qpu_results.json",
        help="completed IBM QPU result JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FIGURE_OUTPUT_DIR,
        help="destination for fig_ibm_validation.pdf and .png",
    )
    args = parser.parse_args(argv)

    ibm = read_json(
        args.input_file,
        {"experiment", "runtime", "circuit", "exact_statevector", "qpu", "agreement_to_exact"},
    )
    configure_style()
    ibm_validation_figure(ibm, args.output_dir, args.output_dir)
    print(f"Generated {args.output_dir / 'fig_ibm_validation.pdf'}")


if __name__ == "__main__":
    main()
