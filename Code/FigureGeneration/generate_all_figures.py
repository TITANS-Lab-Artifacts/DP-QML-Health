#!/usr/bin/env python3
"""Convenience entry point that runs all four independent figure scripts."""

from generate_summaries import main as generate_summaries
from plot_ablation import main as plot_ablation
from plot_clipping import main as plot_clipping
from plot_ibm_validation import main as plot_ibm_validation
from plot_privacy_utility import main as plot_privacy_utility


def main() -> None:
    # Empty argument lists make every child use the documented project paths.
    plot_privacy_utility([])
    plot_clipping([])
    plot_ablation([])
    plot_ibm_validation([])
    generate_summaries([])


if __name__ == "__main__":
    main()
