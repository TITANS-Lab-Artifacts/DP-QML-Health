"""Analyze results without treating repeated budgets as independent seeds."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import qpriv_utils as qp


def paired_bootstrap_ci(
    differences: np.ndarray,
    *,
    draws: int = 10_000,
    alpha: float = 0.05,
    seed: int = 2026,
) -> tuple[float, float]:
    values = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(samples, [alpha / 2, 1 - alpha / 2]))


def safe_wilcoxon(differences: np.ndarray) -> float:
    values = np.asarray(differences, dtype=float)
    if len(values) < 2 or np.allclose(values, 0):
        return np.nan
    return float(wilcoxon(values, alternative="two-sided", method="auto").pvalue)


def holm_adjust(p_values: list[float]) -> list[float]:
    adjusted = [np.nan] * len(p_values)
    valid = [
        (index, value) for index, value in enumerate(p_values) if not np.isnan(value)
    ]
    valid.sort(key=lambda item: item[1])
    running = 0.0
    total = len(valid)
    for rank, (original_index, value) in enumerate(valid):
        running = max(running, (total - rank) * value)
        adjusted[original_index] = min(1.0, running)
    return adjusted


def paired_rows(
    private: pd.DataFrame,
    qnn_model: str,
    classical_model: str,
) -> pd.DataFrame:
    left = private[private.model == qnn_model][
        ["dataset", "eps_target", "seed", "auc"]
    ].rename(columns={"auc": "auc_qnn"})
    right = private[private.model == classical_model][
        ["dataset", "eps_target", "seed", "auc"]
    ].rename(columns={"auc": "auc_classical"})
    pairs = left.merge(
        right,
        on=["dataset", "eps_target", "seed"],
        validate="one_to_one",
    )
    pairs["difference"] = pairs["auc_qnn"] - pairs["auc_classical"]
    return pairs


def corrected_statistics(
    private: pd.DataFrame,
    qnn_model: str,
    classical_model: str,
) -> pd.DataFrame:
    pairs = paired_rows(private, qnn_model, classical_model)
    rows = []
    for (dataset, epsilon), group in pairs.groupby(["dataset", "eps_target"]):
        differences = group.sort_values("seed").difference.to_numpy()
        low, high = paired_bootstrap_ci(differences)
        rows.append(
            {
                "dataset": dataset,
                "scope": "single_budget",
                "eps_target": epsilon,
                "n_independent_seeds": group.seed.nunique(),
                "mean_difference": differences.mean(),
                "ci_low": low,
                "ci_high": high,
                "wilcoxon_p": safe_wilcoxon(differences),
            }
        )

    # One independent curve summary per seed; budgets are averaged within seed.
    for dataset, group in pairs.groupby("dataset"):
        by_seed = group.groupby("seed").difference.mean().to_numpy()
        low, high = paired_bootstrap_ci(by_seed)
        rows.append(
            {
                "dataset": dataset,
                "scope": "curve_average_clustered_by_seed",
                "eps_target": "all",
                "n_independent_seeds": group.seed.nunique(),
                "mean_difference": by_seed.mean(),
                "ci_low": low,
                "ci_high": high,
                "wilcoxon_p": safe_wilcoxon(by_seed),
            }
        )
    result = pd.DataFrame(rows)
    result["holm_p"] = holm_adjust(result["wilcoxon_p"].tolist())
    return result


def main(demo: bool = False) -> None:
    config = qp.load_json(qp.BASE_DIR / "config.json")
    private_path = qp.RESULTS_DIR / (
        "private_results_demo.json" if demo else "private_results.json"
    )
    if not private_path.exists():
        raise FileNotFoundError(f"run private training first: missing {private_path}")
    nonprivate_path = qp.RESULTS_DIR / (
        "nonprivate_results_demo.json" if demo else "nonprivate_results.json"
    )
    if not nonprivate_path.exists():
        raise FileNotFoundError(
            f"run non-private training first: missing {nonprivate_path}"
        )

    private = pd.DataFrame(qp.load_json(private_path))
    nonprivate = pd.DataFrame(qp.load_json(nonprivate_path))
    if set(private.code_version) != {qp.QPRIV_VERSION}:
        raise ValueError("private results mix code versions")

    private_summary = (
        private.groupby(["dataset", "model", "eps_target"])
        .auc.agg(["mean", "std", "count"])
        .reset_index()
    )
    private_summary.insert(0, "training", "private")
    nonprivate_summary = (
        nonprivate.groupby(["dataset", "model"])
        .auc.agg(["mean", "std", "count"])
        .reset_index()
    )
    nonprivate_summary.insert(0, "training", "nonprivate")
    nonprivate_summary["eps_target"] = np.nan
    summary = pd.concat(
        [nonprivate_summary, private_summary],
        ignore_index=True,
    )
    qp.save_records(
        summary.to_dict(orient="records"),
        qp.RESULTS_DIR / "model_summary.json",
        ["training", "dataset", "model", "eps_target"],
    )

    primary = config["primary_comparison"]
    statistics = corrected_statistics(
        private,
        primary["qnn"],
        primary["classical"],
    )
    qp.save_records(
        statistics.to_dict(orient="records"),
        qp.RESULTS_DIR / "corrected_statistical_tests.json",
        ["dataset", "scope", "eps_target"],
    )

    realized = private[
        [
            "dataset",
            "model",
            "eps_target",
            "realized_eps",
            "delta",
            "sigma",
            "sample_rate",
            "steps",
            "n_private_units",
            "privacy_unit",
            "randomness",
            "academic_simulation",
        ]
    ].sort_values(["dataset", "model", "eps_target"])
    qp.save_records(
        realized.to_dict(orient="records"),
        qp.RESULTS_DIR / "privacy_accounting.json",
        ["dataset", "model", "eps_target"],
    )
    print(statistics.round(4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    main(demo=args.demo)
