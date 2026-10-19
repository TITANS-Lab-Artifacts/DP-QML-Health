"""Download, verify, and raw-clean the datasets for Unit-Aware Differentially
Private Quantum Machine Learning for Clinical and Wearable Health Data.

Missing-value imputation is deliberately absent here. It is fitted later on the
designated public calibration partition, never on the private train or test set.
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import qpriv_utils as qp


SOURCES = {
    "pima": {
        "url": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
        "pima-indians-diabetes.data.csv",
        "sha256": "6bfe5d0f379d17a0e0819b996407e3c09bf80febd4287f2ed212190dfff154af",
    },
    "heart": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "heart-disease/processed.cleveland.data",
        "sha256": "a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8",
    },
    "heart_failure": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "00519/heart_failure_clinical_records_dataset.csv",
        "sha256": "9c73cea7468ff5d517801ec050fe9993da5912fce4b56f296f8df3b38dd75912",
    },
}


DEFAULT_CONFIG = {
    "code_version": qp.QPRIV_VERSION,
    "randomness_protocol_version": qp.RANDOMNESS_PROTOCOL_VERSION,
    "datasets": ["pima", "heart", "heart_failure", "wesad"],
    "eps_grid": [0.5, 1, 2, 4, 8],
    "delta": 1e-5,
    "seeds": [0, 1, 2, 3, 4],
    "split": {
        "public_size": 0.2,
        "test_size": 0.2,
        "public_seed": 2026,
    },
    "qnn": {
        "n_features": 4,
        "n_layers": 2,
        "encoding": "ry",
        "device": "lightning.qubit",
        "diff_method": "adjoint",
    },
    "parallel": {
        "start_method": "spawn",
        "maximum_workers": 8,
        "reserved_cpu_cores": 2,
        "threads_per_worker": 1,
    },
    "training": {
        "nonprivate_epochs": 25,
        "private_epochs": 25,
        "expected_batch_units": 32,
        "lr_candidates": {
            "qnn": [0.02, 0.05, 0.08],
            "mlp": [0.01, 0.03, 0.05, 0.08],
            "logistic": [0.01, 0.03, 0.05, 0.08],
        },
    },
    "private_models": [
        "qnn_C0.5_empirical",
        "qnn_C1.0_empirical",
        "qnn_Cbound_certified",
        "mlp_C0.5",
        "mlp_C1.0",
        "logistic_C1.0",
    ],
    "primary_comparison": {
        "qnn": "qnn_C1.0_empirical",
        "classical": "mlp_C1.0",
        "selection": "predeclared_before_private_test_evaluation",
    },
    "ablation": {
        "n_features": [4, 6, 8],
        "n_layers": [1, 2, 3],
        "encodings": ["ry", "rx"],
        "eps": 1.0,
    },
}


def download_verified(name: str, raw_dir: Path) -> Path:
    spec = SOURCES[name]
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / f"{name}.csv"
    if destination.exists() and qp.sha256_file(destination) == spec["sha256"]:
        return destination
    temporary = destination.with_suffix(".download")
    urllib.request.urlretrieve(spec["url"], temporary)
    actual = qp.sha256_file(temporary)
    if actual != spec["sha256"]:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"{name} SHA-256 mismatch: expected {spec['sha256']}, got {actual}"
        )
    os.replace(temporary, destination)
    return destination


def prepare_clinical(raw_dir: Path) -> list[dict]:
    pima_path = download_verified("pima", raw_dir)
    pima_columns = [
        "pregnancies",
        "glucose",
        "bp",
        "skin",
        "insulin",
        "bmi",
        "pedigree",
        "age",
        "target",
    ]
    pima = pd.read_csv(pima_path, header=None, names=pima_columns)
    for column in ["glucose", "bp", "skin", "insulin", "bmi"]:
        pima[column] = pima[column].replace(0, np.nan)
    pima.to_csv(qp.DATA_DIR / "pima_clean.csv", index=False)

    heart_path = download_verified("heart", raw_dir)
    heart_columns = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "num",
    ]
    heart = pd.read_csv(
        heart_path,
        header=None,
        names=heart_columns,
        na_values="?",
    )
    heart["target"] = (heart["num"] > 0).astype(int)
    heart = heart.drop(columns=["num"])
    heart.to_csv(qp.DATA_DIR / "heart_clean.csv", index=False)

    heart_failure_path = download_verified("heart_failure", raw_dir)
    heart_failure = pd.read_csv(heart_failure_path).rename(
        columns={"DEATH_EVENT": "target"}
    )
    heart_failure.to_csv(qp.DATA_DIR / "heart_failure_clean.csv", index=False)

    frames = {
        "pima": pima,
        "heart": heart,
        "heart_failure": heart_failure,
    }
    return [
        {
            "dataset": name,
            "samples": len(frame),
            "features": frame.shape[1] - 1,
            "positives": int(frame["target"].sum()),
            "positive_rate": float(frame["target"].mean()),
            "missing_values_preserved": int(frame.isna().sum().sum()),
            "source_sha256": SOURCES[name]["sha256"],
        }
        for name, frame in frames.items()
    ]


def wesad_features(path: Path) -> tuple[pd.DataFrame, dict]:
    rows = []
    file_hashes = {}
    subject_pattern = re.compile(r"s(\d+)", flags=re.IGNORECASE)
    subjects = sorted(
        (
            directory
            for directory in path.iterdir()
            if directory.is_dir() and subject_pattern.fullmatch(directory.name)
        ),
        key=lambda directory: int(subject_pattern.fullmatch(directory.name).group(1)),
    )
    for subject_dir in subjects:
        subject = subject_dir.name
        pickle_candidates = [
            candidate
            for candidate in subject_dir.glob("*.pkl")
            if candidate.stem.casefold() == subject.casefold()
        ]
        if not pickle_candidates:
            print(f"skip {subject_dir}: matching subject pickle was not found")
            continue
        source = pickle_candidates[0]
        file_hashes[subject] = qp.sha256_file(source)
        with open(source, "rb") as handle:
            payload = pickle.load(handle, encoding="latin1")
        bvp = payload["signal"]["wrist"]["BVP"].ravel()
        eda = payload["signal"]["wrist"]["EDA"].ravel()
        temperature = payload["signal"]["wrist"]["TEMP"].ravel()
        acceleration = payload["signal"]["wrist"]["ACC"]
        labels = payload["label"].ravel().astype(int)
        window_seconds = 60
        n_windows = int(len(labels) / 700 / window_seconds)
        for window in range(n_windows):
            start_label = window * window_seconds * 700
            segment = labels[start_label : start_label + window_seconds * 700]
            counts = np.bincount(segment)
            baseline = counts[1] / len(segment) if len(counts) > 1 else 0.0
            stress = counts[2] / len(segment) if len(counts) > 2 else 0.0
            if baseline > 0.9:
                target = 0
            elif stress > 0.9:
                target = 1
            else:
                continue
            bvp_window = bvp[
                window * window_seconds * 64 : (window + 1) * window_seconds * 64
            ]
            eda_window = eda[
                window * window_seconds * 4 : (window + 1) * window_seconds * 4
            ]
            temp_window = temperature[
                window * window_seconds * 4 : (window + 1) * window_seconds * 4
            ]
            acc_window = np.linalg.norm(
                acceleration[
                    window * window_seconds * 32 : (window + 1) * window_seconds * 32
                ],
                axis=1,
            )
            rows.append(
                {
                    "bvp_mean": bvp_window.mean(),
                    "bvp_std": bvp_window.std(),
                    "eda_mean": eda_window.mean(),
                    "eda_std": eda_window.std(),
                    "temp_mean": temp_window.mean(),
                    "acc_mean": acc_window.mean(),
                    "acc_std": acc_window.std(),
                    "subject": subject,
                    "target": target,
                }
            )
    return pd.DataFrame(rows), file_hashes


def main(wesad_path: Optional[str] = None, trust_wesad_pickle: bool = True) -> None:
    # Preserve user-edited experiment parameters.  A default configuration is
    # created only when the file is missing.
    config_path = qp.BASE_DIR / "config.json"
    if not config_path.exists():
        qp.save_json(DEFAULT_CONFIG, config_path)
    summary = prepare_clinical(qp.DATA_DIR / "raw")
    manifest = {
        name: {"url": spec["url"], "sha256": spec["sha256"]}
        for name, spec in SOURCES.items()
    }

    selected_wesad_path = None
    if wesad_path:
        selected_wesad_path = Path(wesad_path).expanduser()
        if not selected_wesad_path.is_absolute():
            selected_wesad_path = qp.PROJECT_ROOT / selected_wesad_path
    else:
        # Optional raw WESAD data belongs with the other inputs, not in Code/.
        default_wesad_path = qp.DATA_DIR / "WESAD"
        if default_wesad_path.is_dir():
            selected_wesad_path = default_wesad_path

    if selected_wesad_path:
        if not trust_wesad_pickle:
            raise ValueError(
                f"WESAD was found at {selected_wesad_path}. Its files are executable "
                "pickle data. Re-run with --trust-wesad-pickle only after obtaining "
                "the dataset from the official source and verifying the download."
            )
        print(f"loading WESAD from {selected_wesad_path}")
        wesad, hashes = wesad_features(selected_wesad_path)
        if wesad.empty:
            raise ValueError("no usable WESAD windows were found")
        wesad.to_csv(qp.DATA_DIR / "wesad_clean.csv", index=False)
        summary.append(
            {
                "dataset": "wesad",
                "samples": len(wesad),
                "features": wesad.shape[1] - 2,
                "positives": int(wesad["target"].sum()),
                "positive_rate": float(wesad["target"].mean()),
                "subjects": int(wesad["subject"].nunique()),
                "privacy_unit": "subject",
            }
        )
        manifest["wesad"] = {
            "source": "official WESAD download supplied by user",
            "subject_pickle_sha256": hashes,
        }

    summary_frame = pd.DataFrame(summary)
    qp.save_json(summary, qp.RESULTS_DIR / "dataset_info.json")
    qp.save_csv(summary_frame, qp.RESULTS_DIR / "dataset_info.csv")
    qp.save_json(manifest, qp.DATA_DIR / "dataset_manifest.json")
    qp.save_json(qp.environment_fingerprint(), qp.RESULTS_DIR / "environment.json")
    print(summary_frame)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wesad-path",
        help="optional override; defaults to Data/Simulation/WESAD",
    )
    parser.add_argument(
        "--trust-wesad-pickle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "trust WESAD pickle files (default); use "
            "--no-trust-wesad-pickle to refuse loading them"
        ),
    )
    args = parser.parse_args()
    main(args.wesad_path, args.trust_wesad_pickle)
