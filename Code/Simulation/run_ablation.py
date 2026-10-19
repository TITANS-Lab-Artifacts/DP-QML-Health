"""Run the predeclared private QNN ablation with safe parallel workers."""

from __future__ import annotations

import argparse
import itertools
from typing import Optional

import qpriv_utils as qp
from run_nonprivate import available_datasets
from run_private import _run_tasks, derived_seed


def _ablation_task(task: tuple[str, int, int, int, str, dict, bool]) -> dict:
    """Run one feature/depth/encoding configuration in a spawned worker."""

    dataset, seed, n_features, n_layers, encoding, config, demo = task
    qp.configure_worker()
    prepared = qp.split_preprocess(
        qp.load_clean(dataset),
        n_features=n_features,
        split_seed=seed,
        public_seed=config["split"]["public_seed"],
        public_size=config["split"]["public_size"],
        test_size=config["split"]["test_size"],
    )
    if prepared.X_train.shape[1] != n_features:
        return {
            "status": "skipped",
            "dataset": dataset,
            "seed": seed,
            "n_features": n_features,
            "n_layers": n_layers,
            "encoding": encoding,
            "reason": (
                f"requested {n_features} features but only "
                f"{prepared.X_train.shape[1]} are available"
            ),
            "code_version": qp.QPRIV_VERSION,
        }

    def factory():
        return qp.QNN(
            n_features,
            n_layers,
            encoding=encoding,
            diff_method=config["qnn"]["diff_method"],
            device_name=config["qnn"]["device"],
        )

    candidates = config["training"]["lr_candidates"]["qnn"]
    tune_epochs = 20
    epochs = config["training"]["private_epochs"]
    if demo:
        candidates = candidates[:2]
        tune_epochs = 1
        epochs = 1
    learning_rate = qp.tune_lr_on_public_data(
        factory,
        prepared.X_public,
        prepared.y_public,
        candidates,
        seed=seed,
        epochs=tune_epochs,
    )
    key = (dataset, seed, n_features, n_layers, encoding)
    sampling_seed = derived_seed("ablation-sample", *key)
    noise_seed = derived_seed("ablation-noise", *key)
    if demo:
        X_private = prepared.X_train[:64]
        y_private = prepared.y_train[:64]
        units_private = prepared.unit_ids_train[:64]
    else:
        X_private = prepared.X_train
        y_private = prepared.y_train
        units_private = prepared.unit_ids_train
    model, privacy = qp.train_dp_manual(
        factory,
        X_private,
        y_private,
        unit_ids=units_private,
        target_eps=config["ablation"]["eps"],
        delta=config["delta"],
        C=1.0,
        epochs=epochs,
        lr=learning_rate,
        expected_batch_units=config["training"]["expected_batch_units"],
        init_seed=seed,
        sampling_seed=sampling_seed,
        noise_seed=noise_seed,
    )
    artifact_path = (
        qp.ARTIFACT_DIR
        / "private_ablation"
        / dataset
        / f"features{n_features}_layers{n_layers}_{encoding}_seed{seed}.pt"
    )
    qp.save_model_state(
        model,
        artifact_path,
        {
            "dataset": dataset,
            "seed": seed,
            "n_features": n_features,
            "n_layers": n_layers,
            "encoding": encoding,
            "eps_target": config["ablation"]["eps"],
            "C": 1.0,
            "quantum_device": config["qnn"]["device"],
            "diff_method": config["qnn"]["diff_method"],
            "randomness": "separately_seeded_torch_generators",
            "randomness_protocol_version": qp.RANDOMNESS_PROTOCOL_VERSION,
            "academic_simulation": True,
            "code_version": qp.QPRIV_VERSION,
        },
    )
    return {
        "status": "completed",
        "dataset": dataset,
        "seed": seed,
        "n_features": n_features,
        "n_layers": n_layers,
        "encoding": encoding,
        "eps_target": config["ablation"]["eps"],
        "C": 1.0,
        "threshold_status": "predeclared_empirical",
        "qnn_certificate": qp.qnn_gradient_certificate(model),
        "selected_lr_public": learning_rate,
        "params": qp.count_params(model),
        "model_artifact": qp.project_relative(artifact_path),
        "quantum_device": config["qnn"]["device"],
        "diff_method": config["qnn"]["diff_method"],
        "code_version": qp.QPRIV_VERSION,
        "randomness_protocol_version": qp.RANDOMNESS_PROTOCOL_VERSION,
        **prepared.split_info,
        **privacy,
        **qp.public_gradient_diagnostics(
            model,
            prepared.X_public,
            prepared.y_public,
            1.0,
            unit_ids=prepared.unit_ids_public,
        ),
        **qp.evaluate(model, prepared.X_test, prepared.y_test),
    }


def _key(row: dict) -> tuple:
    return (
        row["dataset"],
        int(row["seed"]),
        int(row["n_features"]),
        int(row["n_layers"]),
        row["encoding"],
    )


def main(
    demo: bool = False,
    progress_callback=None,
    workers: Optional[int] = None,
) -> None:
    """Run missing ablations and checkpoint JSON/CSV only in the parent."""

    config = qp.load_json(qp.BASE_DIR / "config.json")
    grid = config["ablation"]
    datasets = available_datasets(config)
    seeds = config["seeds"]
    combinations = list(
        itertools.product(
            grid["n_features"],
            grid["n_layers"],
            grid["encodings"],
        )
    )
    if demo:
        datasets = datasets[:1]
        seeds = seeds[:1]
        combinations = combinations[:1]

    suffix = "_demo" if demo else ""
    results_path = qp.RESULTS_DIR / f"ablation_results{suffix}.json"
    skipped_path = qp.RESULTS_DIR / f"ablation_skipped{suffix}.json"
    results = qp.load_json(results_path, default=[])
    results = [row for row in results if row.get("code_version") == qp.QPRIV_VERSION]
    skipped = qp.load_json(skipped_path, default=[])
    skipped = [row for row in skipped if row.get("code_version") == qp.QPRIV_VERSION]
    done = {_key(row) for row in results} | {_key(row) for row in skipped}
    all_tasks = [
        (dataset, seed, n_features, n_layers, encoding, config, demo)
        for dataset, seed, (n_features, n_layers, encoding) in itertools.product(
            datasets,
            seeds,
            combinations,
        )
    ]
    pending = [task for task in all_tasks if task[:5] not in done]
    completed = len(all_tasks) - len(pending)
    for index in range(completed):
        if progress_callback:
            progress_callback(index + 1, len(all_tasks), "ablation runs (resumed)")
    effective_workers = qp.worker_count(workers, len(pending))
    print(
        f"ablation stage: {len(pending)} pending job(s), {effective_workers} worker(s)",
        flush=True,
    )

    for row in _run_tasks(pending, _ablation_task, effective_workers):
        completed += 1
        if row["status"] == "skipped":
            skipped.append(row)
            qp.save_records(
                skipped,
                skipped_path,
                ["dataset", "seed", "n_features", "n_layers", "encoding"],
            )
            detail = f"skipped: {row['reason']}"
        else:
            results.append(row)
            qp.save_records(
                results,
                results_path,
                ["dataset", "seed", "n_features", "n_layers", "encoding"],
            )
            detail = (
                f"{row['dataset']} {row['n_features']}/{row['n_layers']}/"
                f"{row['encoding']} AUC={row['auc']:.3f}"
            )
        print(detail, flush=True)
        if progress_callback:
            progress_callback(completed, len(all_tasks), f"ablation runs ({detail})")

    qp.save_records(
        results,
        results_path,
        ["dataset", "seed", "n_features", "n_layers", "encoding"],
    )
    qp.save_records(
        skipped,
        skipped_path,
        ["dataset", "seed", "n_features", "n_layers", "encoding"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    main(demo=args.demo, workers=args.workers)
