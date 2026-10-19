"""Shared plotting functions for the four quantitative result figures.

This is a support module, not a command-line program.  Each ``plot_*.py``
script loads only the result files required for one figure and calls one
function from this module.  Keeping common styles here makes the four outputs
consistent without duplicating hundreds of lines of plotting code.

Outputs written to Results/Figures/Generated:
    fig_privacy_utility.pdf   privacy utility curves for the matched comparison
    fig_clipping.pdf          clipping threshold sensitivity and noise scale
    fig_ablation.pdf          encoding, component count, and depth ablation
    fig_ibm_validation.pdf    IBM QPU agreement with exact statevector inference
    paper_*.csv, paper_results_summary.json  numeric backing for the manuscript
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# ----------------------------------------------------------------------------
# Shared naming and styling
# ----------------------------------------------------------------------------

DATASET_ORDER = ["pima", "heart", "heart_failure", "wesad"]
DATASET_LABELS = {
    "pima": "Pima Diabetes",
    "heart": "UCI Heart",
    "heart_failure": "Heart Failure",
    "wesad": "WESAD",
}
DATASET_SHORT = {
    "pima": "Pima",
    "heart": "UCI Heart",
    "heart_failure": "Heart Fail.",
    "wesad": "WESAD",
}

PRIMARY_ORDER = ["qnn_C1.0_empirical", "mlp_C1.0", "logistic_C1.0"]
PRIMARY_LABELS = {
    "qnn_C1.0_empirical": "QNN ($C{=}1$)",
    "mlp_C1.0": "MLP ($C{=}1$)",
    "logistic_C1.0": "Logistic ($C{=}1$)",
}
PRIMARY_COLORS = {
    "qnn_C1.0_empirical": "#1F4E9C",
    "mlp_C1.0": "#D9741A",
    "logistic_C1.0": "#2E8B57",
}
PRIMARY_MARKERS = {
    "qnn_C1.0_empirical": "o",
    "mlp_C1.0": "s",
    "logistic_C1.0": "^",
}

CLIP_ORDER = ["qnn_C0.5_empirical", "qnn_C1.0_empirical", "qnn_Cbound_certified"]
CLIP_LABELS = {
    "qnn_C0.5_empirical": r"$C=0.5$ (empirical)",
    "qnn_C1.0_empirical": r"$C=1$ (empirical)",
    "qnn_Cbound_certified": r"$C=\sqrt{8}$ (certified)",
}
CLIP_VALUES = {
    "qnn_C0.5_empirical": 0.5,
    "qnn_C1.0_empirical": 1.0,
    "qnn_Cbound_certified": float(np.sqrt(8.0)),
}
CLIP_COLORS = {
    "qnn_C0.5_empirical": "#8FB8E0",
    "qnn_C1.0_empirical": "#2E75B6",
    "qnn_Cbound_certified": "#0B2545",
}

NONPRIVATE_MAP = {
    "qnn_C1.0_empirical": "qnn",
    "mlp_C1.0": "mlp_matched",
    "logistic_C1.0": "logistic_torch",
}

EPS_GRID = [0.5, 1.0, 2.0, 4.0, 8.0]
TEXT_WIDTH = 7.16
GRID_COLOR = "#D3D3D3"

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parents[1]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Nimbus Roman",
                "Times New Roman",
                "TeX Gyre Termes",
                "Liberation Serif",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.8,
            "legend.fontsize": 8.3,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, mirror_dir: Path, stem: str) -> None:
    metadata = {
        "Title": stem.replace("_", " "),
        "Author": "Unit-Aware DP-QML Health reproducibility script",
        "Subject": "Unit-Aware Differentially Private Quantum Machine Learning for Clinical and Wearable Health Data: version 4 experimental results",
    }
    # ``mirror_dir`` is retained for compatibility with the original plotting
    # functions. In this project both paths normally point to the
    # one canonical Results/Figures/Generated directory.
    directories = {output_dir.resolve(), mirror_dir.resolve()}
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    # The LaTeX project receives vector PDF only, which keeps the Overleaf
    # upload small. The mirror directory receives PDF and a high resolution PNG
    # for slides, posters, and quick inspection.
    for directory in directories:
        fig.savefig(
            directory / f"{stem}.pdf",
            bbox_inches="tight",
            pad_inches=0.015,
            metadata=metadata,
        )
        fig.savefig(
            directory / f"{stem}.png",
            dpi=400,
            bbox_inches="tight",
            pad_inches=0.015,
        )
    plt.close(fig)


def style_axis(axis: plt.Axes, y_grid_only: bool = False) -> None:
    axis.grid(
        True,
        axis="y" if y_grid_only else "both",
        color=GRID_COLOR,
        linewidth=0.5,
        zorder=0,
    )
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color("#4A4A4A")


# ----------------------------------------------------------------------------
# Figure: privacy utility curves
# ----------------------------------------------------------------------------


def privacy_utility_figure(
    private: pd.DataFrame,
    nonprivate: pd.DataFrame,
    output_dir: Path,
    mirror_dir: Path,
) -> None:
    summary = (
        private[private["model"].isin(PRIMARY_ORDER)]
        .groupby(["dataset", "model", "eps_target"], as_index=False)["auc"]
        .agg(mean="mean", std="std")
    )
    reference = nonprivate.groupby(["dataset", "model"], as_index=False)["auc"].mean()

    fig, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH, 3.55), sharex=True, sharey=True)
    for panel, (axis, dataset) in enumerate(zip(axes.flat, DATASET_ORDER)):
        style_axis(axis)
        base = float(
            reference[
                (reference["dataset"] == dataset)
                & (reference["model"] == NONPRIVATE_MAP["qnn_C1.0_empirical"])
            ]["auc"].iloc[0]
        )
        axis.axhline(base, color="#666666", linewidth=1.0, linestyle=(0, (5, 3)), zorder=1)
        axis.axhline(0.5, color="#B0B0B0", linewidth=0.9, linestyle=(0, (1, 2)), zorder=1)

        for model in PRIMARY_ORDER:
            rows = summary[
                (summary["dataset"] == dataset) & (summary["model"] == model)
            ].sort_values("eps_target")
            x = rows["eps_target"].to_numpy(dtype=float)
            mean = rows["mean"].to_numpy(dtype=float)
            std = rows["std"].to_numpy(dtype=float)
            axis.fill_between(
                x,
                np.clip(mean - std, 0.0, 1.0),
                np.clip(mean + std, 0.0, 1.0),
                color=PRIMARY_COLORS[model],
                alpha=0.13,
                linewidth=0,
                zorder=2,
            )
            axis.plot(
                x,
                mean,
                color=PRIMARY_COLORS[model],
                marker=PRIMARY_MARKERS[model],
                linewidth=1.6,
                markersize=4.6,
                markeredgecolor="white",
                markeredgewidth=0.5,
                zorder=3,
            )

        axis.set_xscale("log", base=2)
        axis.set_xticks(EPS_GRID)
        axis.set_xticklabels(["0.5", "1", "2", "4", "8"])
        axis.set_xlim(0.42, 9.8)
        axis.set_ylim(0.22, 1.0)
        axis.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        axis.set_title(
            f"({chr(97 + panel)}) {DATASET_LABELS[dataset]}", pad=4, fontweight="bold"
        )
        axis.text(
            0.975,
            0.045,
            f"nonprivate QNN  {base:.3f}",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.6,
            color="#3A3A3A",
        )

    for row in range(2):
        axes[row, 0].set_ylabel("Test ROC AUC")
    for column in range(2):
        axes[1, column].set_xlabel(r"Target privacy budget $\varepsilon$")

    handles = [
        Line2D(
            [],
            [],
            color=PRIMARY_COLORS[model],
            marker=PRIMARY_MARKERS[model],
            markersize=5,
            markeredgecolor="white",
            linewidth=1.6,
            label=PRIMARY_LABELS[model],
        )
        for model in PRIMARY_ORDER
    ]
    handles.append(
        Line2D(
            [],
            [],
            color="#666666",
            linewidth=1.0,
            linestyle=(0, (5, 3)),
            label="Nonprivate QNN",
        )
    )
    handles.append(
        Line2D(
            [],
            [],
            color="#B0B0B0",
            linewidth=1.0,
            linestyle=(0, (1, 2)),
            label="Random ranking",
        )
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 1.042),
        columnspacing=1.8,
        handletextpad=0.55,
    )
    fig.subplots_adjust(
        top=0.875, bottom=0.115, left=0.062, right=0.995, hspace=0.28, wspace=0.06
    )
    save_figure(fig, output_dir, mirror_dir, "fig_privacy_utility")


# ----------------------------------------------------------------------------
# Figure: clipping threshold sensitivity
# ----------------------------------------------------------------------------


def clipping_figure(private: pd.DataFrame, output_dir: Path, mirror_dir: Path) -> None:
    curve = (
        private[private["model"].isin(CLIP_ORDER)]
        .groupby(["dataset", "model", "seed"], as_index=False)["auc"]
        .mean()
        .groupby(["dataset", "model"], as_index=False)["auc"]
        .agg(mean="mean", std="std")
    )
    noise = (
        private[private["eps_target"] == 1.0]
        .groupby(["dataset", "model"], as_index=False)["sigma"]
        .first()
    )

    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.70))
    positions = np.arange(len(DATASET_ORDER), dtype=float)
    width = 0.25

    axis = axes[0]
    style_axis(axis, y_grid_only=True)
    for index, model in enumerate(CLIP_ORDER):
        offset = (index - 1) * width
        means, stds = [], []
        for dataset in DATASET_ORDER:
            row = curve[(curve["dataset"] == dataset) & (curve["model"] == model)]
            means.append(float(row["mean"].iloc[0]))
            stds.append(float(row["std"].iloc[0]))
        bars = axis.bar(
            positions + offset,
            means,
            width * 0.92,
            yerr=stds,
            capsize=2.2,
            color=CLIP_COLORS[model],
            edgecolor="white",
            linewidth=0.5,
            error_kw={"elinewidth": 0.8, "ecolor": "#3A3A3A"},
            label=CLIP_LABELS[model],
            zorder=3,
        )
        for bar, value in zip(bars, means):
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                0.03,
                f"{value:.3f}",
                ha="left",
                va="bottom",
                rotation=90,
                fontsize=7.0,
                color="white",
                zorder=4,
            )
    axis.axhline(0.5, color="#7A7A7A", linewidth=0.9, linestyle=(0, (1, 2)), zorder=2)
    axis.set_xticks(positions)
    axis.set_xticklabels([DATASET_SHORT[d] for d in DATASET_ORDER])
    axis.set_ylim(0.0, 1.02)
    axis.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axis.set_ylabel("Curve average test ROC AUC")
    axis.set_xlabel("Dataset")
    axis.set_title("(a) Utility against clipping threshold", pad=4, fontweight="bold")

    axis = axes[1]
    style_axis(axis, y_grid_only=True)
    for index, model in enumerate(CLIP_ORDER):
        offset = (index - 1) * width
        values = []
        for dataset in DATASET_ORDER:
            sigma = float(
                noise[(noise["dataset"] == dataset) & (noise["model"] == model)][
                    "sigma"
                ].iloc[0]
            )
            values.append(sigma * CLIP_VALUES[model])
        bars = axis.bar(
            positions + offset,
            values,
            width * 0.92,
            color=CLIP_COLORS[model],
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                value * 1.10,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=6.9,
                color="#2A2A2A",
                zorder=4,
            )
    axis.set_yscale("log")
    axis.set_ylim(1.0, 260.0)
    axis.set_xticks(positions)
    axis.set_xticklabels([DATASET_SHORT[d] for d in DATASET_ORDER])
    axis.set_ylabel(r"Noise scale $\sigma C$ at $\varepsilon = 1$")
    axis.set_xlabel("Dataset")
    axis.set_title("(b) Injected noise scale per coordinate", pad=4, fontweight="bold")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.045),
        columnspacing=2.0,
        handlelength=1.4,
        handletextpad=0.5,
    )
    fig.subplots_adjust(left=0.070, right=0.995, top=0.80, bottom=0.155, wspace=0.24)
    save_figure(fig, output_dir, mirror_dir, "fig_clipping")


# ----------------------------------------------------------------------------
# Figure: circuit ablation
# ----------------------------------------------------------------------------


def ablation_figure(ablation: pd.DataFrame, output_dir: Path, mirror_dir: Path) -> None:
    panels = [
        ("encoding", ["ry", "rx"], [r"$R_Y$", r"$R_X$"], "Encoding gate"),
        ("n_features", [4, 6, 8], ["4", "6", "8"], "PCA components"),
        ("n_layers", [1, 2, 3], ["1", "2", "3"], "Variational layers"),
    ]
    shades = ["#9DC3E6", "#2E75B6", "#0B2545"]

    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.60), sharey=True)
    positions = np.arange(len(DATASET_ORDER), dtype=float)

    for panel, (axis, (column, values, labels, xlabel)) in enumerate(zip(axes, panels)):
        style_axis(axis, y_grid_only=True)
        grouped = ablation.groupby(["dataset", column], as_index=False)["auc"].agg(
            mean="mean", std="std"
        )
        width = 0.78 / len(values)
        for index, value in enumerate(values):
            offset = (index - (len(values) - 1) / 2.0) * width
            means, stds, xs = [], [], []
            for slot, dataset in enumerate(DATASET_ORDER):
                row = grouped[
                    (grouped["dataset"] == dataset) & (grouped[column] == value)
                ]
                if row.empty:
                    continue
                means.append(float(row["mean"].iloc[0]))
                stds.append(float(row["std"].iloc[0]))
                xs.append(positions[slot] + offset)
            axis.bar(
                xs,
                means,
                width * 0.90,
                yerr=stds,
                capsize=1.9,
                color=shades[index],
                edgecolor="white",
                linewidth=0.5,
                error_kw={"elinewidth": 0.75, "ecolor": "#3A3A3A"},
                label=labels[index],
                zorder=3,
            )
        axis.axhline(0.5, color="#7A7A7A", linewidth=0.9, linestyle=(0, (1, 2)), zorder=2)
        axis.set_xticks(positions)
        axis.set_xticklabels(
            [DATASET_SHORT[d] for d in DATASET_ORDER], rotation=20, ha="right"
        )
        axis.set_xlabel("Dataset")
        axis.set_ylim(0.0, 1.02)
        axis.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
        axis.set_title(f"({chr(97 + panel)}) {xlabel}", pad=4, fontweight="bold")
        axis.legend(
            loc="upper center",
            ncol=len(values),
            frameon=False,
            fontsize=7.9,
            handlelength=1.2,
            handletextpad=0.4,
            columnspacing=1.0,
            borderpad=0.1,
            labelspacing=0.25,
        )

    axes[0].set_ylabel(r"Test ROC AUC at $\varepsilon = 1$")
    fig.subplots_adjust(left=0.065, right=0.995, top=0.875, bottom=0.235, wspace=0.08)
    save_figure(fig, output_dir, mirror_dir, "fig_ablation")


# ----------------------------------------------------------------------------
# Figure: IBM QPU agreement
# ----------------------------------------------------------------------------


def ibm_validation_figure(ibm: dict, output_dir: Path, mirror_dir: Path) -> None:
    exact = ibm["exact_statevector"]
    qpu = ibm["qpu"]
    agreement = ibm["agreement_to_exact"]
    shots = int(ibm["circuit"]["shots_per_binding"])
    labels = ibm["experiment"]["labels"]

    series = (
        ("nonprivate", "Nonprivate checkpoint", "Nonprivate", "#1F4E9C", "o"),
        (
            "private",
            r"Private checkpoint, $\varepsilon{=}1$",
            r"Private, $\varepsilon{=}1$",
            "#B23A48",
            "s",
        ),
    )

    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.70))

    axis = axes[0]
    style_axis(axis)
    limit = 0.80
    axis.plot([0, limit], [0, limit], color="#4A4A4A", linewidth=1.0, zorder=2)
    axis.text(
        0.615,
        0.655,
        "perfect agreement",
        rotation=45,
        rotation_mode="anchor",
        fontsize=7.2,
        color="#3A3A3A",
        ha="center",
        va="bottom",
    )
    for name, _long_label, label, color, marker in series:
        x = np.asarray(exact[f"{name}_probabilities"], dtype=float)
        y = np.asarray(qpu[f"{name}_probabilities"], dtype=float)
        error = 1.96 * np.sqrt(np.clip(y * (1.0 - y) / shots, 0.0, None))
        axis.errorbar(
            x,
            y,
            yerr=error,
            color=color,
            marker=marker,
            markersize=5.0,
            markeredgecolor="white",
            markeredgewidth=0.5,
            linestyle="none",
            capsize=2.2,
            elinewidth=0.8,
            zorder=3,
            label=f"{label}, MAE $=$ {agreement[name]['mae']:.3f}",
        )
    axis.axvline(0.5, color="#B0B0B0", linewidth=0.8, linestyle=(0, (1, 2)), zorder=1)
    axis.axhline(0.5, color="#B0B0B0", linewidth=0.8, linestyle=(0, (1, 2)), zorder=1)
    axis.set_xlim(0, limit)
    axis.set_ylim(0, limit)
    axis.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
    axis.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Exact statevector probability")
    axis.set_ylabel("IBM QPU probability")
    axis.set_title("(a) Hardware against exact inference", pad=4, fontweight="bold")
    axis.legend(
        loc="lower right",
        frameon=False,
        fontsize=7.4,
        handletextpad=0.4,
        borderpad=0.2,
        labelspacing=0.32,
    )

    axis = axes[1]
    style_axis(axis, y_grid_only=True)
    cases = np.arange(len(labels), dtype=float)
    width = 0.37
    for index, (name, long_label, _short_label, color, _marker) in enumerate(series):
        x = np.asarray(exact[f"{name}_probabilities"], dtype=float)
        y = np.asarray(qpu[f"{name}_probabilities"], dtype=float)
        axis.bar(
            cases + (index - 0.5) * width,
            y - x,
            width * 0.90,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            label=long_label,
            zorder=3,
        )
    band = 1.96 * float(np.sqrt(0.25 / shots))
    band_patch = axis.axhspan(
        -band, band, color="#BBBBBB", alpha=0.45, zorder=1, label="95% shot interval"
    )
    axis.axhline(0.0, color="#4A4A4A", linewidth=0.8, zorder=2)
    axis.set_xticks(cases)
    axis.set_xticklabels(
        [f"c{i + 1}\n$y{{=}}${labels[i]}" for i in range(len(labels))], fontsize=7.6
    )
    axis.set_xlim(-0.62, 7.62)
    axis.set_xlabel("Fixed Pima test case")
    axis.set_ylabel("QPU minus exact probability")
    axis.set_ylim(-0.08, 0.19)
    axis.set_yticks([-0.04, 0.00, 0.04, 0.08, 0.12, 0.16])
    axis.set_title("(b) Signed error for each case", pad=4, fontweight="bold")
    handles, legend_labels = axis.get_legend_handles_labels()
    if band_patch not in handles:
        handles.append(band_patch)
        legend_labels.append("95% shot interval")
    axis.legend(
        handles,
        legend_labels,
        loc="upper right",
        ncol=1,
        frameon=False,
        fontsize=7.4,
        handlelength=1.2,
        handletextpad=0.4,
        borderpad=0.1,
        labelspacing=0.3,
    )

    fig.subplots_adjust(left=0.070, right=0.995, top=0.885, bottom=0.185, wspace=0.24)
    save_figure(fig, output_dir, mirror_dir, "fig_ibm_validation")


# ----------------------------------------------------------------------------
# Numeric summaries used by the manuscript
# ----------------------------------------------------------------------------


def write_summaries(
    nonprivate: pd.DataFrame,
    private: pd.DataFrame,
    corrected_tests: pd.DataFrame,
    ablation: pd.DataFrame,
    ibm: dict,
    mirror_dir: Path,
) -> None:
    """Write the numeric backing of the manuscript into the mirror directory."""
    nonprivate_summary = nonprivate.groupby(["dataset", "model"], as_index=False)[
        "auc"
    ].agg(mean="mean", std="std", count="count")
    private_summary = private.groupby(
        ["dataset", "model", "eps_target"], as_index=False
    )["auc"].agg(mean="mean", std="std", count="count")
    curve_summary = (
        private.groupby(["dataset", "model", "seed"], as_index=False)["auc"]
        .mean()
        .groupby(["dataset", "model"], as_index=False)["auc"]
        .agg(mean="mean", std="std", count="count")
    )
    ablation_summary = ablation.groupby(
        ["dataset", "n_features", "n_layers", "encoding"], as_index=False
    )["auc"].agg(mean="mean", std="std", count="count")
    accounting = private.groupby(["dataset", "eps_target"], as_index=False).agg(
        realized_eps=("realized_eps", "max"),
        sigma=("sigma", "first"),
        sample_rate=("sample_rate", "first"),
        steps=("steps", "first"),
        private_units=("private_units", "first"),
    )

    nonprivate_summary.insert(0, "analysis", "nonprivate_auc")
    private_summary.insert(0, "analysis", "private_auc")
    curve_summary.insert(0, "analysis", "private_curve_average_auc")
    combined = pd.concat(
        [
            nonprivate_summary.rename(columns={"model": "configuration"}),
            private_summary.rename(columns={"model": "configuration"}),
            curve_summary.rename(columns={"model": "configuration"}),
        ],
        ignore_index=True,
        sort=False,
    )

    payload = {
        "source": "Unit-Aware Differentially Private Quantum Machine Learning for Clinical and Wearable Health Data: version 4 experiment results",
        "completed_runs": {
            "nonprivate": int(len(nonprivate)),
            "private": int(len(private)),
            "ablation": int(len(ablation)),
            "total": int(len(nonprivate) + len(private) + len(ablation)),
        },
        "maximum_target_to_realized_gap": float(
            (private["eps_target"] - private["realized_eps"]).abs().max()
        ),
        "nonprivate_auc": nonprivate_summary.to_dict(orient="records"),
        "private_auc": private_summary.to_dict(orient="records"),
        "private_curve_auc": curve_summary.to_dict(orient="records"),
        "privacy_accounting": accounting.to_dict(orient="records"),
        "corrected_tests": corrected_tests.to_dict(orient="records"),
        "ablation_auc": ablation_summary.to_dict(orient="records"),
        "ibm": {
            "backend": ibm["runtime"]["backend"],
            "job_id": ibm["runtime"]["job_id"],
            "quantum_seconds": ibm["runtime"]["job_metrics"]["usage"]["quantum_seconds"],
            "shots_per_binding": ibm["circuit"]["shots_per_binding"],
            "parameter_bindings": ibm["circuit"]["parameter_bindings"],
            "logical_depth": ibm["circuit"]["logical_depth"],
            "isa_depth": ibm["circuit"]["isa_depth"],
            "isa_size": ibm["circuit"]["isa_size"],
            "agreement_to_exact": ibm["agreement_to_exact"],
            "exact_statevector": ibm["exact_statevector"],
            "qpu": ibm["qpu"],
        },
    }

    for directory in {mirror_dir.resolve()}:
        directory.mkdir(parents=True, exist_ok=True)
        combined.to_csv(directory / "paper_model_summary.csv", index=False)
        ablation_summary.to_csv(directory / "paper_ablation_summary.csv", index=False)
        corrected_tests.to_csv(directory / "paper_corrected_tests.csv", index=False)
        accounting.to_csv(directory / "paper_privacy_accounting.csv", index=False)
        with (directory / "paper_results_summary.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
